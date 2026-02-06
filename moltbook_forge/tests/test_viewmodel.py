from __future__ import annotations

from moltbook_forge.ui.viewmodel import NexusViewModelBuilder
from moltbook_forge.ui.evidence import EvidenceResolver


def test_viewmodel_cache_hits(tmp_path):
    resolver = EvidenceResolver(cache_path=str(tmp_path / "cache.json"))
    builder = NexusViewModelBuilder(resolver, cache={})
    urls = ["https://github.com/eidosian-forge/core"]
    content = "Title Content"
    summary, _ = builder.build_evidence_summary(urls, content)
    assert summary.url_count == 1
    summary2, _ = builder.build_evidence_summary(urls, content)
    assert summary2.url_count == 1
    assert builder.stats["summary_hits"] == 1
    assert builder.stats["summary_misses"] == 1
