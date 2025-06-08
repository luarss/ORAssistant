from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class UserInput(BaseModel):
    """User input model for chat queries."""
    query: str = Field(..., description="User's natural language query", min_length=1)
    chat_history: List[Dict[str, str]] = Field(
        default=[], 
        description="Previous conversation history"
    )
    list_sources: bool = Field(
        default=False, 
        description="Whether to include source URLs in response"
    )
    list_context: bool = Field(
        default=False, 
        description="Whether to include context excerpts in response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is operations research?",
                "chat_history": [
                    {"User": "Hello", "AI": "Hi! How can I help you today?"}
                ],
                "list_sources": True,
                "list_context": True
            }
        }


class ContextSource(BaseModel):
    """Source and context information for a response."""
    source: str = Field(default="", description="URL or reference to the source")
    context: str = Field(default="", description="Relevant context excerpt from source")

    class Config:
        json_schema_extra = {
            "example": {
                "source": "https://example.com/operations-research",
                "context": "Operations research is a discipline that applies analytical methods..."
            }
        }


class ChatResponse(BaseModel):
    """Standard chat response model."""
    response: str = Field(..., description="Generated response text")
    context_sources: List[ContextSource] = Field(
        default=[], 
        description="Sources and context used for the response"
    )
    tools: List[str] = Field(
        default=[], 
        description="Tools or agents used to generate the response"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Operations research is a mathematical science...",
                "context_sources": [
                    {
                        "source": "https://example.com/or-intro",
                        "context": "OR applies mathematical and scientific methods..."
                    }
                ],
                "tools": ["retriever", "reranker"]
            }
        }


class ChatToolResponse(BaseModel):
    """Alternative response model with separate source and context lists."""
    response: str = Field(..., description="Generated response text")
    sources: List[str] = Field(default=[], description="List of source URLs")
    context: List[str] = Field(default=[], description="List of context excerpts")
    tools: List[str] = Field(default=[], description="Tools used in generation")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    message: str = Field(default="API is healthy", description="Status message")
    version: str = Field(default="1.0.0", description="API version")
    timestamp: Optional[str] = Field(default=None, description="Timestamp of health check")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok",
                "message": "API is healthy and operational", 
                "version": "1.0.0",
                "timestamp": "2025-06-08T15:30:00Z"
            }
        }
