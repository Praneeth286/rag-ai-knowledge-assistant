"""FastAPI service exposing the RAG knowledge assistant."""
from __future__ import annotations

import asyncio

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag

app = FastAPI(title="RAG Knowledge Assistant", version="1.0.0")

# Loaded lazily so the app can boot before an index exists.
_store = None


def _get_store():
    global _store
    if _store is None:
        _store = rag.load_store()
    return _store


class AskRequest(BaseModel):
    question: str


class IngestRequest(BaseModel):
    folder: str = "data"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(req: IngestRequest) -> dict:
    global _store
    try:
        _store = rag.ingest(req.folder)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"status": "indexed", "folder": req.folder}


@app.post("/ask")
def ask(req: AskRequest) -> dict:
    try:
        return rag.answer(req.question, store=_get_store())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/ask/stream")
async def ask_stream(req: AskRequest) -> StreamingResponse:
    """Stream the answer token-by-token as Server-Sent Events."""
    result = await asyncio.to_thread(rag.answer, req.question, _get_store())

    async def event_gen():
        for word in result["answer"].split():
            yield f"data: {word} \n\n"
            await asyncio.sleep(0.02)
        yield f"event: sources\ndata: {', '.join(result['sources'])}\n\n"

    return StreamingResponse(event_gen(), media_type="text/event-stream")
