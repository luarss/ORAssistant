"""
Configuration management for ORAssistant API.

This module handles all environment variable loading and validation,
providing centralized configuration for the application.
"""

import os
from typing import Optional, List
from pydantic_settings import BaseSettings
from pydantic import Field


class APISettings(BaseSettings):
    """API configuration settings."""
    
    # Server settings
    backend_host: str = Field(default="0.0.0.0", alias="BACKEND_URL")
    backend_port: int = Field(default=8000, alias="BACKEND_PORT") 
    backend_workers: int = Field(default=1, alias="BACKEND_WORKERS")
    backend_reload: bool = Field(default=False, alias="BACKEND_RELOAD")
    
    # Logging settings
    log_level: str = Field(default="INFO", alias="LOGLEVEL")
    
    # CORS settings
    cors_origins: List[str] = Field(default=["*"], alias="CORS_ORIGINS")
    
    # API metadata
    api_title: str = "ORAssistant API"
    api_description: str = "AI-powered research and question-answering system using RAG"
    api_version: str = "1.0.0"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"  # Ignore extra environment variables


class LLMSettings(BaseSettings):
    """LLM and AI model configuration settings."""
    
    # Core LLM settings
    use_cuda: bool = Field(default=False, alias="USE_CUDA")
    llm_temp: float = Field(default=0.0, alias="LLM_TEMP")
    fast_mode: bool = Field(default=False, alias="FAST_MODE")
    llm_model: str = Field(default="", alias="LLM_MODEL")
    
    # Model-specific settings
    ollama_model: Optional[str] = Field(default=None, alias="OLLAMA_MODEL")
    google_gemini: Optional[str] = Field(default=None, alias="GOOGLE_GEMINI")
    
    # Embeddings settings
    embeddings_type: str = Field(default="", alias="EMBEDDINGS_TYPE")
    hf_embeddings: Optional[str] = Field(default=None, alias="HF_EMBEDDINGS")
    google_embeddings: Optional[str] = Field(default=None, alias="GOOGLE_EMBEDDINGS")
    
    # Reranking settings
    hf_reranker: str = Field(default="", alias="HF_RERANKER")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instances - only create if env vars are set
api_settings = APISettings()

try:
    llm_settings = LLMSettings()
except Exception as e:
    # Create with defaults if environment variables are not set
    llm_settings = None


def get_llm_settings() -> LLMSettings:
    """Get LLM settings, validating required variables."""
    validate_required_env_vars()
    return LLMSettings()


def validate_required_env_vars() -> None:
    """Validate that all required environment variables are set."""
    required_vars = [
        "USE_CUDA",
        "LLM_TEMP", 
        "HF_EMBEDDINGS",
        "HF_RERANKER",
        "LLM_MODEL",
    ]
    
    missing_vars = [var for var in required_vars if os.getenv(var) is None]
    if missing_vars:
        raise ValueError(
            f"The following environment variables are not set: {', '.join(missing_vars)}"
        )
