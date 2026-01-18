"""FastAPI service exposing EidosCore operations."""

from __future__ import annotations

import os
from wsgiref.simple_server import make_server
from typing import Callable

from fastapi import FastAPI

from core.eidos_core import EidosCore
from core.health import HealthChecker

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ENABLE_UI = os.getenv("ENABLE_UI", "false").lower() == "true"

app = FastAPI(title="Eidos API", docs_url="/docs" if ENABLE_UI else None)
core = EidosCore()


def create_app(checker: HealthChecker | None = None) -> Callable:
    """Return a WSGI app exposing a ``/healthz`` endpoint."""

    checker = checker or HealthChecker()

    def wsgi_app(environ: dict, start_response: Callable) -> list[bytes]:
        if environ.get("PATH_INFO") == "/healthz":
            start_response("200 OK", [("Content-Type", "application/json")])
            return [b'{"status": "ok"}']
        start_response("404 Not Found", [])
        return [b""]

    return wsgi_app


def run_server() -> None:
    """Launch the WSGI server using :func:`create_app`."""

    with make_server(HOST, PORT, create_app()) as server:
        server.serve_forever()


@app.get("/memories")
def get_memories() -> list[object]:
    """Return stored memories."""
    return core.reflect()


@app.post("/remember")
def add_memory(experience: str) -> dict[str, str]:
    """Store a new experience."""
    core.remember(experience)
    return {"status": "stored"}


@app.post("/recurse")
def run_recurse() -> dict[str, str]:
    """Run recursion on current memories."""
    core.recurse()
    return {"status": "recurred"}


@app.post("/process")
def process(experience: str) -> dict[str, str]:
    """Remember an experience and immediately recurse."""
    core.process_cycle(experience)
    return {"status": "processed"}


def main() -> None:
    """Launch the API server."""
    import uvicorn

    uvicorn.run("api.server:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()
