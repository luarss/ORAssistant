import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging

from ..config import api_settings
from .routers import graphs, healthcheck

# Configure logging
logging.basicConfig(
    level=api_settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info("ORAssistant API starting up...")
    logger.info(f"API Version: {api_settings.api_version}")
    yield
    logger.info("ORAssistant API shutting down...")


# Create FastAPI app with enhanced configuration
app = FastAPI(
    title=api_settings.api_title,
    description=api_settings.api_description,
    version=api_settings.api_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=api_settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Global HTTP exception handler."""
    logger.error(f"HTTP error occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "status_code": exc.status_code}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled error occurred: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "status_code": 500}
    )


# Include routers
app.include_router(healthcheck.router, tags=["health"])
app.include_router(graphs.router, tags=["graphs"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to ORAssistant API",
        "version": api_settings.api_version,
        "title": api_settings.api_title,
        "docs": "/docs",
        "health": "/healthcheck"
    }
