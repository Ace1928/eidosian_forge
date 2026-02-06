from __future__ import annotations

from datetime import datetime

from moltbook_forge.client import MoltbookPost
from moltbook_forge.pipeline import SignalPipeline


def test_pipeline_deduplicates() -> None:
    pipeline = SignalPipeline()
    post_a = MoltbookPost(
        id="p1",
        title="Eidos Forge Protocol",
        content="Eidos forge protocol verification",
        author="tester",
        timestamp=datetime.now(),
    )
    post_b = MoltbookPost(
        id="p2",
        title="Eidos Forge Protocol",
        content="Eidos forge protocol verification",
        author="tester",
        timestamp=datetime.now(),
    )
    results = pipeline.process_batch([post_a, post_b])
    assert len(results) == 1
