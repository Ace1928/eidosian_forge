import pytest
from fastapi.testclient import TestClient
from moltbook_forge.ui.app import app
from moltbook_forge.client import MockMoltbookClient
from moltbook_forge.interest import InterestEngine

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_dashboard_status(client):
    """Test that the dashboard loads correctly."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Moltbook Nexus" in response.text
    assert "Eidosian" in response.text

def test_post_comments_partial(client):
    """Test that the comments partial returns content."""
    # Mock post ID from mock client is p0, p1, etc.
    response = client.get("/post/p1")
    assert response.status_code == 200
    assert "Agreed." in response.text

def test_interest_scoring():
    """Test the interest engine logic."""
    engine = InterestEngine()
    from moltbook_forge.client import MoltbookPost
    from datetime import datetime
    
    post = MoltbookPost(
        id="test",
        content="This is a post about recursive Eidos intelligence and the Forge.",
        author="tester",
        timestamp=datetime.now(),
        url="http://test.com"
    )
    breakdown = engine.analyze_post(post)
    assert breakdown.total > 20
    assert "eidos" in [k.lower() for k in breakdown.matched_keywords]

def test_mock_client_data():
    """Test that the mock client returns data."""
    import asyncio
    mock = MockMoltbookClient()
    posts = asyncio.run(mock.get_posts())
    assert len(posts) > 0
    # Updated mock client uses EidosianForge or CipherSTW
    assert posts[0].author in ["EidosianForge", "CipherSTW"]

def test_dashboard_benchmark(benchmark, client):
    """Benchmark the dashboard endpoint."""
    result = benchmark(client.get, "/")
    assert result.status_code == 200