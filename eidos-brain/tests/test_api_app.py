import httpx
import pytest

from labs.api_app import create_app


@pytest.mark.anyio
async def test_api_remember_and_reflect() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/remember", json={"value": "hi"})
        assert response.json()["status"] == "stored"
        memories = (await client.get("/reflect")).json()["memories"]
        assert "hi" in memories


@pytest.mark.anyio
async def test_api_recurse() -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        await client.post("/remember", json={"value": "data"})
        count = (await client.post("/recurse")).json()["count"]
        assert count == 2
