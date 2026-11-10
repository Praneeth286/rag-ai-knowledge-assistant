"""Single-process demo API. Put authentication in front before shared deployment."""
from contextlib import aclosing
import asyncio
import json
import logging
from uuid import UUID, uuid4
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import rag
from sessions import Sessions, SessionBusy

app = FastAPI(title='RAG Knowledge Assistant', version='1.1.0')
_store = None
_store_lock = asyncio.Lock()
sessions = Sessions()
log = logging.getLogger(__name__)

async def _get_store():
    global _store
    async with _store_lock:
        if _store is None:
            try:
                _store = await asyncio.to_thread(rag.load_store)
            except FileNotFoundError as exc:
                raise HTTPException(409, 'Index missing; ingest documents first') from exc
            except ValueError as exc:
                raise HTTPException(409, 'Invalid index; re-ingest documents') from exc
        return _store

class AskRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    question: str = Field(min_length=1, max_length=4000)
    session_id: UUID | None = None

class IngestRequest(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    folder: str = Field(default='data', min_length=1, max_length=512)

    @field_validator('folder')
    @classmethod
    def no_null(cls, value):
        if '\x00' in value:
            raise ValueError('Invalid folder')
        return value

@app.get('/health')
def health():
    return {'status': 'ok'}

@app.post('/ingest')
async def ingest(req: IngestRequest):
    global _store
    async with _store_lock:
        try:
            _store = await asyncio.to_thread(rag.ingest, req.folder)
        except (ValueError, FileNotFoundError, UnicodeError) as exc:
            raise HTTPException(400, 'Invalid document folder or documents') from exc
        except Exception as exc:
            log.exception('Ingestion failed')
            raise HTTPException(502, 'Ingestion failed') from exc
    return {'status': 'indexed'}

@app.post('/ask')
async def ask(req: AskRequest):
    store = await _get_store()
    key = str(req.session_id or uuid4())
    try:
        async with sessions.use(key) as history:
            result = await rag.answer_async(req.question, store, list(history))
            sessions.commit(history, req.question, result['answer'])
            return {**result, 'session_id': key}
    except SessionBusy as exc:
        raise HTTPException(429, str(exc)) from exc
    except Exception as exc:
        log.exception('Generation failed')
        raise HTTPException(502, 'Generation failed') from exc

def event(kind, data):
    return f'event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n'

@app.post('/ask/stream')
async def ask_stream(req: AskRequest, request: Request):
    store = await _get_store()
    key = str(req.session_id or uuid4())

    async def generate():
        try:
            async with sessions.use(key) as history:
                yield event('session', {'session_id': key})
                parts = []
                async with aclosing(rag.stream_answer(req.question, store, list(history))) as stream:
                    async for kind, value in stream:
                        if await request.is_disconnected():
                            return
                        if kind == 'token':
                            parts.append(value)
                        yield event(kind, value)
                if await request.is_disconnected():
                    return
                sessions.commit(history, req.question, ''.join(parts))
                yield event('done', {})
        except asyncio.CancelledError:
            raise
        except SessionBusy:
            yield event('error', {'message': 'Session capacity reached; retry later'})
        except Exception:
            log.exception('Streaming generation failed')
            yield event('error', {'message': 'Generation failed; retry the request'})

    return StreamingResponse(generate(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
