# RAG Knowledge Assistant

A FastAPI learning project for document Q&A with LangChain, OpenAI and FAISS.
It supports model-output streaming, session history and source citations. This is a
single-process demo, not a production deployment.

## Setup

Use Python 3.11+ in a virtual environment:

```sh
python -m venv .venv
# Activate .venv (Windows: .venv\Scripts\Activate.ps1)
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. Never commit credentials.
Add UTF-8 `.txt`/`.md` files to the `data` directory.

```sh
python rag.py --ingest ./data
uvicorn main:app --reload
python -m pytest -q
```

Open http://localhost:8000/docs for the API schema. `/health` is a liveness check;
it does not verify the provider credentials or index readiness.

## Conversation API

POST `/ask` with `{"question":"What is the refund policy?"}`. The response contains
`answer`, `sources`, and a server-generated UUID `session_id`. Send that ID in later
requests to `/ask` or `/ask/stream` to continue the conversation. Omit it to start a
new conversation. Questions are trimmed, limited to 4,000 characters, and must be
nonempty. Invalid input returns 422. A missing or invalid index returns 409.
Provider failures return a generic 502 without exposing credentials or error details.

Memory holds the last 10 successful question/answer pairs. It expires after one hour
of inactivity and permits at most 256 sessions. Requests for the same session are
serialized; separate sessions have separate histories. Failed or interrupted streams
do not append partial turns. History is used in retrieval and in the model messages.
Memory disappears on restart and is not shared across workers. Capacity errors return
429 for `/ask`, or an SSE error event for streaming requests.

## Streaming protocol

```sh
curl -N http://localhost:8000/ask/stream -H 'Content-Type: application/json' \
  -d '{"question":"What is the refund policy?"}'
```

The POST endpoint uses Server-Sent Events. Use a fetch streaming reader or an SSE
client supporting POST; browser EventSource cannot POST. Each event has JSON data:

```text
event: session
data: {"session_id":"<uuid>"}

event: token
data: "Text including spaces and escaped\nnewlines"

event: sources
data: ["policy.md"]

event: done
data: {}
```

`token` events forward actual model output chunks as they arrive; chunks may contain
multiple tokens. There is no artificial word splitting or delay. Decode the JSON and
concatenate token strings exactly. After headers are sent, failures produce `error`
with a generic `message`, without `done`. Treat only `done` as successful completion.
Disconnects close the upstream generator; provider billing may continue briefly.
Disable proxy buffering for streaming. Provider timeout is 60 seconds with one retry;
answers are capped at 2,048 output tokens.

## Index and ingestion security

POST `/ingest` with `{"folder":"data"}` rebuilds the configured index. Folders must
resolve under `DATA_ROOT` (default `data`). Only nonempty UTF-8 text/Markdown files
are accepted. Limits: 2 MB per file, 20 MB total, 1,000 files. External symlink targets
are rejected. Do not allow untrusted users to modify the source tree during ingestion.

Indexes are saved atomically as `INDEX_PATH/index.json`, containing text, source names,
embedding vectors, format version and embedding model. Loading validates structure,
size, dimensions and finite numbers, then reconstructs FAISS in memory. No pickle or
native FAISS index is deserialized. Old `index.pkl`/`index.faiss` files are ignored:
**re-ingest your source documents after upgrading**. JSON is limited to 100 MB on load;
this format trades startup speed and disk space for a simpler, data-only boundary.
Keep the index directory writable only by the service operator. Validation does not
authenticate document contents or prevent document poisoning.

## Deployment limits and verification

There is no authentication, tenant isolation, rate limiting or cloud deployment in this
project. Session IDs are bearer identifiers, not authorization. Before exposing the API,
add authentication/authorization (especially for ingestion), enforce request limits at
the proxy, protect logs, and use a shared session store for multiple workers. Documents,
questions and conversation history are sent to the configured OpenAI service.
Retrieved documents can contain prompt injection; prompt instructions reduce risk but
are not a security boundary. Citations do not guarantee answer correctness.

Tests use deterministic fake embeddings and model streams, plus real FAISS construction
and search. They cover request validation, session isolation/bounds, streaming text and
ordering, incremental model consumption, failure behavior, safe index round-trips and
malformed/legacy index rejection. They make no paid provider calls. Live OpenAI latency,
answer quality, real-network cancellation, proxy streaming and AWS deployment require
separate integration verification. Do not claim these are tested from unit tests alone.
