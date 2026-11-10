from functools import lru_cache
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', extra='ignore')
    openai_api_key: str = ''
    chat_model: str = 'gpt-4o'
    embedding_model: str = 'text-embedding-3-small'
    chunk_size: int = Field(800, ge=100, le=10000)
    chunk_overlap: int = Field(120, ge=0)
    top_k: int = Field(4, ge=1, le=20)
    index_path: str = 'faiss_index'
    data_root: str = 'data'

    @model_validator(mode='after')
    def validate_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError('chunk_overlap must be smaller than chunk_size')
        return self

@lru_cache
def get_settings():
    return Settings()
