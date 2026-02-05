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
    assert "Moltbook" in response.text
    assert "Nexus" in response.text
    assert "EidosianForge" in response.text

def test_post_comments_partial(client):
    """Test that the comments partial returns content."""
    response = client.get("/post/p1")
    assert response.status_code == 200
    assert "User" in response.text

def test_interest_scoring():
    """Test the interest engine logic."""
    engine = InterestEngine()
    from moltbook_forge.client import MoltbookPost
    from datetime import datetime
    
    post = MoltbookPost(
        id="test",
        content="This is a post about Eidos and the Forge.",
        author="tester",
        timestamp=datetime.now(),
        url="http://test.com"
    )
    score = engine.score_post(post)
    assert score > 15  # Eidos(10) + Forge(8)

def test_mock_client_data():
    """Test that the mock client returns data."""
    import asyncio
    mock = MockMoltbookClient()
    posts = asyncio.run(mock.get_posts())
    assert len(posts) > 0
    assert posts[0].author == "EidosAgent"
