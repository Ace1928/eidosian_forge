import httpx
import pytest

from labs.api_demo import app


@pytest.mark.anyio
async def test_remember_and_list() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/remember", json={"item": "hello"})
        assert response.status_code == 200
        assert response.json()["count"] >= 1

        response = await client.get("/memories")
        assert response.status_code == 200
        assert "hello" in response.json()
