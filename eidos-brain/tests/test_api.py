import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import httpx
import pytest

from api.server import app, core


@pytest.mark.anyio
async def test_remember_endpoint() -> None:
    """Store a memory via the API."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        core.memory.clear()
        response = await client.post("/remember", params={"experience": "hello"})
        assert response.status_code == 200
        assert response.json()["status"] == "stored"
        assert core.memory == ["hello"]


@pytest.mark.anyio
async def test_recurse_endpoint() -> None:
    """Run recursion through the API."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        core.memory = ["hi"]
        response = await client.post("/recurse")
        assert response.status_code == 200
        assert len(core.memory) == 2


@pytest.mark.anyio
async def test_process_endpoint() -> None:
    """Combine remembering and recursion."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        core.memory.clear()
        response = await client.post("/process", params={"experience": "x"})
        assert response.status_code == 200
        assert core.memory[0] == "x"
        assert len(core.memory) == 2


@pytest.mark.anyio
async def test_get_memories() -> None:
    """Retrieve stored memories."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        core.memory = ["a", {"repr": "'a'"}]
        response = await client.get("/memories")
        assert response.status_code == 200
        assert response.json() == core.memory


@pytest.mark.anyio
async def test_healthz_endpoint() -> None:
    """Expose a basic health check."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/healthz")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


@pytest.mark.anyio
async def test_readyz_endpoint() -> None:
    """Expose a readiness check."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"
