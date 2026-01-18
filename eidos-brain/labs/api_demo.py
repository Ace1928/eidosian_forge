"""Simple REST API exposing EidosCore functionality."""

from fastapi import FastAPI
from pydantic import BaseModel

from core.eidos_core import EidosCore

app = FastAPI(title="Eidos API")
core = EidosCore()


class MemoryItem(BaseModel):
    item: str


@app.post("/remember")
def remember(payload: MemoryItem) -> dict[str, int]:
    """Store ``payload.item`` in memory and return the count."""
    core.remember(payload.item)
    return {"count": len(core.memory)}


@app.get("/memories")
def memories() -> list:
    """Return all stored memories."""
    return core.reflect()
