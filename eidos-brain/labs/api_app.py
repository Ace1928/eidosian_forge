"""FastAPI server exposing EidosCore operations."""

from fastapi import FastAPI
from pydantic import BaseModel

from core.eidos_core import EidosCore


class Experience(BaseModel):
    """Payload for new experiences."""

    value: str


def create_app() -> FastAPI:
    """Return an API with memory management routes."""
    app = FastAPI(title="Eidos API")
    app.state.core = EidosCore()

    @app.post("/remember")
    def remember(exp: Experience) -> dict[str, str]:
        """Store an experience in memory."""
        app.state.core.remember(exp.value)
        return {"status": "stored"}

    @app.get("/reflect")
    def reflect() -> dict[str, list]:
        """Return all stored memories."""
        return {"memories": app.state.core.reflect()}

    @app.post("/recurse")
    def recurse() -> dict[str, int]:
        """Run a recursion cycle and return memory count."""
        app.state.core.recurse()
        return {"count": app.state.core.memory_count()}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
