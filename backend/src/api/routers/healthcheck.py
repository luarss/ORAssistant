import os
from datetime import datetime
from fastapi import APIRouter
from ..models.response_model import HealthCheckResponse

router = APIRouter()


@router.get("/healthcheck", response_model=HealthCheckResponse)
async def healthcheck() -> HealthCheckResponse:
    """
    Health check endpoint to verify API status.
    
    Returns:
        HealthCheckResponse: Current health status of the API
    """
    return HealthCheckResponse(
        status="ok", 
        message="API is healthy and operational",
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat() + "Z"
    )
