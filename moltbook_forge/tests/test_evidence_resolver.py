from __future__ import annotations

from moltbook_forge.ui.evidence import EvidenceResolver


def test_domain_scoring_defaults(tmp_path) -> None:
    resolver = EvidenceResolver(cache_path=str(tmp_path / "cache.json"))
    score, label = resolver.score_domain("github.com")
    assert score == 90
    assert label == "high"

    score, label = resolver.score_domain("bit.ly")
    assert score == 20
    assert label == "low"

    score, label = resolver.score_domain("example.com")
    assert score == 50
    assert label == "medium"


def test_resolve_urls_sets_credibility(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MOLTBOOK_EVIDENCE_FETCH", "false")
    resolver = EvidenceResolver(cache_path=str(tmp_path / "cache.json"))
    items = resolver.resolve_urls(["https://github.com/example/repo"])
    assert items[0].credibility_label == "high"
    assert items[0].credibility_score == 90
