from fastapi.testclient import TestClient
from labs.api_demo import app


def test_remember_and_list() -> None:
    client = TestClient(app)
    response = client.post("/remember", json={"item": "hello"})
    assert response.status_code == 200
    assert response.json()["count"] >= 1

    response = client.get("/memories")
    assert response.status_code == 200
    assert "hello" in response.json()
