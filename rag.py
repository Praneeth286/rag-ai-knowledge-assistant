"""Core RAG pipeline: ingestion, chunking, embeddings, retrieval, and generation.

Run as a script to ingest a folder of documents:

    python rag.py --ingest ./data
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import get_settings

settings = get_settings()


def _embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def load_documents(folder: str) -> list[Document]:
    """Load .txt and .md files from a folder into LangChain Documents."""
    docs: list[Document] = []
    for path in Path(folder).rglob("*"):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    if not docs:
        raise FileNotFoundError(f"No .txt or .md files found in {folder!r}")
    return docs


def chunk(docs: Iterable[Document]) -> list[Document]:
    """Split documents into semantically coherent, overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(list(docs))


def ingest(folder: str) -> FAISS:
    """Build (or rebuild) the FAISS index from a folder of documents."""
    chunks = chunk(load_documents(folder))
    store = FAISS.from_documents(chunks, _embeddings())
    store.save_local(settings.index_path)
    print(f"Indexed {len(chunks)} chunks -> {settings.index_path}")
    return store


def load_store() -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    if not os.path.exists(settings.index_path):
        raise FileNotFoundError(
            "No FAISS index found. Run `python rag.py --ingest <folder>` first."
        )
    return FAISS.load_local(
        settings.index_path, _embeddings(), allow_dangerous_deserialization=True
    )


_PROMPT = (
    "You are a precise assistant. Answer the question using ONLY the context below. "
    "If the answer is not in the context, say you don't know. Cite sources by name.\n\n"
    "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
)


def _format_context(docs: list[Document]) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', '?')}] {d.page_content}" for d in docs)


def answer(question: str, store: FAISS | None = None) -> dict:
    """Retrieve relevant chunks and generate a grounded answer with citations."""
    store = store or load_store()
    retrieved = store.similarity_search(question, k=settings.top_k)
    context = _format_context(retrieved)

    llm = ChatOpenAI(
        model=settings.chat_model,
        temperature=0,
        api_key=settings.openai_api_key,
    )
    response = llm.invoke(_PROMPT.format(context=context, question=question))

    sources = sorted({d.metadata.get("source", "?") for d in retrieved})
    return {"answer": response.content, "sources": sources}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG knowledge assistant")
    parser.add_argument("--ingest", metavar="FOLDER", help="Folder of documents to index")
    parser.add_argument("--ask", metavar="QUESTION", help="Ask a question against the index")
    args = parser.parse_args()

    if args.ingest:
        ingest(args.ingest)
    if args.ask:
        result = answer(args.ask)
        print(result["answer"])
        print("\nSources:", ", ".join(result["sources"]))
