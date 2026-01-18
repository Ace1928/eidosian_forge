from fastapi.testclient import TestClient
from labs.api_app import create_app


def test_api_remember_and_reflect() -> None:
    app = create_app()
    client = TestClient(app)
    assert client.post("/remember", json={"value": "hi"}).json()["status"] == "stored"
    memories = client.get("/reflect").json()["memories"]
    assert "hi" in memories


def test_api_recurse() -> None:
    app = create_app()
    client = TestClient(app)
    client.post("/remember", json={"value": "data"})
    count = client.post("/recurse").json()["count"]
    assert count == 2
