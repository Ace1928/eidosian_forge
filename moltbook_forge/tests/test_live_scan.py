from __future__ import annotations

from moltbook_forge.tools.live_scan import summarize_posts


def test_summarize_posts() -> None:
    posts = [
        {"id": "1", "author": "a", "content": "alpha", "score": 10, "risk": "low", "intent": "TECHNICAL", "submolt": "general", "keywords": ["forge"]},
        {"id": "2", "author": "b", "content": "beta", "score": 5, "risk": "low", "intent": "SOCIAL", "submolt": "general", "keywords": ["agent"]},
    ]
    summary = summarize_posts(posts, top_n=1)
    assert summary.total == 2
    assert len(summary.top_posts) == 1
    assert summary.top_posts[0]["id"] == "1"


def test_summarize_posts_benchmark(benchmark) -> None:
    posts = []
    for i in range(200):
        posts.append(
            {
                "id": str(i),
                "author": "agent",
                "content": f"post {i}",
                "score": i % 50,
                "risk": "low",
                "intent": "TECHNICAL",
                "submolt": "general",
                "keywords": ["forge", "agent"],
            }
        )
    result = benchmark(summarize_posts, posts, 5)
    assert result.total == 200
