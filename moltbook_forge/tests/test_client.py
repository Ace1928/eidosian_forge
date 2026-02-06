from __future__ import annotations

from datetime import datetime

from moltbook_forge.client import MoltbookClient


def test_parse_post_content_none() -> None:
    client = MoltbookClient(api_key="test", base_url="https://www.moltbook.com/api/v1")
    raw = {
        "id": "abc",
        "title": "Test",
        "content": None,
        "author": {"name": "tester"},
        "created_at": datetime.now().isoformat(),
        "upvotes": 1,
        "comment_count": 0,
    }
    post = client._parse_post(raw)
    assert post.content == ""
    assert post.author == "tester"
