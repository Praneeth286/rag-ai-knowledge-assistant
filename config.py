"""Application configuration loaded from environment variables."""
from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = ""
    chat_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"

    # Retrieval / chunking
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4

    # Vector store
    index_path: str = "faiss_index"

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
