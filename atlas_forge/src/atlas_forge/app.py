from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import logging

from .config import TEMPLATES_DIR, STATIC_DIR, setup_pythonpath

# Initialize environment
setup_pythonpath()

# --- Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eidos_dashboard")

# --- App Setup ---
app = FastAPI(title="Eidosian Atlas", version="1.1.0")

from .templating import templates

# Mount Static Files
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Routes will be included here
from .routes import dashboard, browser, api

app.include_router(dashboard.router)
app.include_router(browser.router)
app.include_router(api.router)

@app.get("/health")
async def health_check():
    return {"ok": True, "service": "atlas_forge", "status": "healthy"}
