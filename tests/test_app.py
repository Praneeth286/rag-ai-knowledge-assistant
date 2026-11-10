import asyncio
import json
from types import SimpleNamespace
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import main
import rag
from sessions import Sessions, SessionBusy

class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, float(len(t) % 3)] for t in texts]
    def embed_query(self, text):
        return [1.0, 0.0]

@pytest.fixture(autouse=True)
def isolate(monkeypatch):
    monkeypatch.setattr(main, '_store', object())
    monkeypatch.setattr(main, 'sessions', Sessions())
    monkeypatch.setattr(main, '_store_lock', asyncio.Lock())

@pytest.mark.parametrize('payload', [{}, {'question': '  '}, {'question':'x'*4001},
    {'question':'ok','session_id':'bad'}, {'question':'ok','unexpected':1}])
def test_validation(payload):
    assert TestClient(main.app).post('/ask', json=payload).status_code == 422

def test_sessions_and_followup(monkeypatch):
    seen=[]
    async def answer(q, store, history):
        seen.append(history)
        return {'answer':'Answer', 'sources':['a.md']}
    monkeypatch.setattr(rag, 'answer_async', answer)
    with TestClient(main.app) as client:
        first=client.post('/ask',json={'question':'First'}).json()
        client.post('/ask',json={'question':'Follow up','session_id':first['session_id']})
        client.post('/ask',json={'question':'Separate'})
    assert seen == [[], [('human','First'),('ai','Answer')], []]

def test_stream_preserves_text_and_history(monkeypatch):
    async def stream(*args):
        yield 'token', ' A\n'
        yield 'token', '  B!'
        yield 'sources', ['a.md']
    monkeypatch.setattr(rag, 'stream_answer', stream)
    with TestClient(main.app) as client:
        response=client.post('/ask/stream',json={'question':'Q'})
    events=[(x.splitlines()[0][7:],json.loads(x.splitlines()[1][6:]))
            for x in response.text.strip().split('\n\n')]
    assert [k for k,v in events] == ['session','token','token','sources','done']
    assert ''.join(v for k,v in events if k=='token') == ' A\n  B!'
    key=events[0][1]['session_id']
    assert main.sessions.items[key].history[-1] == ('ai',' A\n  B!')

def test_failed_stream_not_committed(monkeypatch):
    async def stream(*args):
        yield 'token', 'partial'
        raise RuntimeError('secret')
    monkeypatch.setattr(rag,'stream_answer',stream)
    response=TestClient(main.app).post('/ask/stream',json={'question':'Q'})
    assert 'event: error' in response.text and 'secret' not in response.text
    assert all(not s.history for s in main.sessions.items.values())

def test_missing_index(monkeypatch):
    monkeypatch.setattr(main,'_store',None)
    def missing():
        raise FileNotFoundError()
    monkeypatch.setattr(rag,'load_store',missing)
    with TestClient(main.app) as client:
        for route in ['/ask','/ask/stream']:
            assert client.post(route,json={'question':'Q'}).status_code == 409

def test_real_pipeline_streams_incrementally(monkeypatch):
    calls=[]
    class Store:
        def similarity_search(self, query, k):
            calls.append(query)
            return [Document(page_content='facts',metadata={'source':'a'})]
    class LLM:
        async def astream(self, messages):
            assert messages[-2] == ('ai','prior answer')
            yield SimpleNamespace(content='first ')
            calls.append('resumed')
            yield SimpleNamespace(content='second')
    monkeypatch.setattr(rag,'_llm',lambda:LLM())
    async def run():
        stream=rag.stream_answer('followup',Store(),[('ai','prior answer')])
        assert await anext(stream) == ('token','first ')
        assert 'resumed' not in calls
        assert await anext(stream) == ('token','second')
        assert await anext(stream) == ('sources',['a'])
    asyncio.run(run())
    assert 'prior answer\nfollowup' in calls

def test_safe_index_roundtrip_and_tamper(tmp_path,monkeypatch):
    root=tmp_path/'docs'; root.mkdir(); (root/'a.md').write_text('refund within 30 days')
    monkeypatch.setattr(rag.settings,'data_root',str(root))
    monkeypatch.setattr(rag.settings,'index_path',str(tmp_path/'index'))
    monkeypatch.setattr(rag,'_embeddings',lambda:FakeEmbeddings())
    rag.ingest(str(root))
    assert rag.load_store().similarity_search('refund',k=1)[0].metadata['source']=='a.md'
    path=tmp_path/'index'/'index.json'
    payload=json.loads(path.read_text()); payload['vectors'][0][0]='bad'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): rag.load_store()
    path.unlink(); (path.parent/'index.pkl').write_bytes(b'never deserialize')
    with pytest.raises(FileNotFoundError): rag.load_store()

def test_ingestion_boundaries(tmp_path,monkeypatch):
    root=tmp_path/'docs'; root.mkdir()
    monkeypatch.setattr(rag.settings,'data_root',str(root))
    with pytest.raises(ValueError): rag.load_documents(str(tmp_path))
    with pytest.raises(FileNotFoundError): rag.load_documents(str(root))
    (root/'large.txt').write_bytes(b'x'*2_000_001)
    with pytest.raises(ValueError): rag.load_documents(str(root))

def test_memory_bounds_serialization_and_expiry():
    async def run():
        sessions=Sessions(capacity=1,ttl=-1,turns=1)
        async with sessions.use('one') as history:
            sessions.commit(history,'q','a'); sessions.commit(history,'q2','a2')
            assert history == [('human','q2'),('ai','a2')]
            with pytest.raises(SessionBusy):
                async with sessions.use('two'): pass
        async with sessions.use('two'): pass
        assert 'one' not in sessions.items
    asyncio.run(run())

def test_same_session_serializes_and_cancel_releases():
    async def run():
        sessions=Sessions()
        entered=asyncio.Event()
        release=asyncio.Event()
        order=[]
        async def first():
            async with sessions.use('one'):
                entered.set()
                await release.wait()
        async def second():
            async with sessions.use('one'):
                order.append('second')
        task=asyncio.create_task(first())
        await entered.wait()
        other=asyncio.create_task(second())
        await asyncio.sleep(0)
        assert not order
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        await other
        assert order == ['second']
        assert sessions.items['one'].users == 0
    asyncio.run(run())

def test_stream_cancellation_closes_model(monkeypatch):
    closed=[]
    async def stream(*args):
        try:
            yield 'token','first'
            await asyncio.Event().wait()
        finally:
            closed.append(True)
    monkeypatch.setattr(rag,'stream_answer',stream)
    async def run():
        class Request:
            async def is_disconnected(self): return False
        response=await main.ask_stream(main.AskRequest(question='Q'),Request())
        iterator=response.body_iterator
        await anext(iterator) # session
        await anext(iterator) # first token
        task=asyncio.create_task(anext(iterator))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError): await task
        assert closed
        assert all(not entry.history and not entry.users for entry in main.sessions.items.values())
    asyncio.run(run())
