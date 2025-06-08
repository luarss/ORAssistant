import os
import logging
import uvicorn
from dotenv import load_dotenv

from src.config import api_settings
from src.api.main import app

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=api_settings.log_level.upper(),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main application entry point."""
    logger.info(f"Starting ORAssistant API on {api_settings.backend_host}:{api_settings.backend_port}")
    logger.info(f"Workers: {api_settings.backend_workers}, Reload: {api_settings.backend_reload}")
    
    uvicorn.run(
        "src.api.main:app",
        host=api_settings.backend_host,
        port=api_settings.backend_port,
        workers=api_settings.backend_workers if not api_settings.backend_reload else 1,  # Can't use multiple workers with reload
        reload=api_settings.backend_reload,
        access_log=True,
        log_level=api_settings.log_level.lower()
    )


if __name__ == "__main__":
    main()
