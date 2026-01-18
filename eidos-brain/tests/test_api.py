import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from api.server import app, core


def test_remember_endpoint() -> None:
    """Store a memory via the API."""
    client = TestClient(app)
    core.memory.clear()
    response = client.post("/remember", params={"experience": "hello"})
    assert response.status_code == 200
    assert response.json()["status"] == "stored"
    assert core.memory == ["hello"]


def test_recurse_endpoint() -> None:
    """Run recursion through the API."""
    client = TestClient(app)
    core.memory = ["hi"]
    response = client.post("/recurse")
    assert response.status_code == 200
    assert len(core.memory) == 2


def test_process_endpoint() -> None:
    """Combine remembering and recursion."""
    client = TestClient(app)
    core.memory.clear()
    response = client.post("/process", params={"experience": "x"})
    assert response.status_code == 200
    assert core.memory[0] == "x"
    assert len(core.memory) == 2


def test_get_memories() -> None:
    """Retrieve stored memories."""
    client = TestClient(app)
    core.memory = ["a", {"repr": "'a'"}]
    response = client.get("/memories")
    assert response.status_code == 200
    assert response.json() == core.memory
