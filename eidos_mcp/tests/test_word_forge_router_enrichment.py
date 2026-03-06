from __future__ import annotations

import json
from pathlib import Path

from eidos_mcp.routers import word_forge


def test_wf_enrich_term_updates_graph(tmp_path: Path, monkeypatch) -> None:
    graph_path = tmp_path / "semantic_graph.json"
    monkeypatch.setattr(word_forge, "SEMANTIC_GRAPH_PATH", graph_path)
    monkeypatch.setattr(word_forge, "_graph", None)
    monkeypatch.setattr(
        word_forge,
        "_generate_structured_payload",
        lambda **_: {
            "definition": "The capacity for self-directed action.",
            "pos": "noun",
            "aliases": ["self-direction"],
            "domains": ["cognition", "agency"],
            "related_terms": [{"term": "agency", "relation_type": "related", "weight": 0.9}],
            "_effective_thinking_mode": "on",
        },
    )

    payload = json.loads(word_forge.wf_enrich_term("autonomy"))
    assert payload["status"] == "success"
    assert payload["effective_thinking_mode"] == "on"

    saved = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = {row["id"]: row for row in saved["nodes"]}
    assert nodes["autonomy"]["definition"] == "The capacity for self-directed action."
    assert "agency" in nodes["autonomy"]["domains"]
    assert any(edge["source"] == "autonomy" and edge["target"] == "agency" for edge in saved["edges"])


def test_wf_build_lexicon_from_text_uses_model_payload(tmp_path: Path, monkeypatch) -> None:
    graph_path = tmp_path / "semantic_graph.json"
    monkeypatch.setattr(word_forge, "SEMANTIC_GRAPH_PATH", graph_path)
    monkeypatch.setattr(word_forge, "_graph", None)
    monkeypatch.setattr(
        word_forge,
        "_generate_structured_payload",
        lambda **_: {
            "terms": [
                {
                    "term": "vector",
                    "definition": "A numerical representation used for semantic retrieval.",
                    "pos": "noun",
                    "domains": ["retrieval"],
                    "related_terms": [{"term": "embedding", "relation_type": "related", "weight": 0.8}],
                },
                {
                    "term": "embedding",
                    "definition": "A learned representation of text.",
                    "pos": "noun",
                    "domains": ["ml"],
                    "related_terms": [],
                },
            ],
            "_effective_thinking_mode": "off",
        },
    )

    payload = json.loads(word_forge.wf_build_lexicon_from_text("Vector retrieval uses embeddings."))
    assert payload["status"] == "success"
    assert payload["nodes_added"] >= 2
    assert payload["edges_added"] >= 1

    stats = json.loads(word_forge.wf_graph_stats())
    assert stats["nodes"] >= 2
    assert stats["edges"] >= 1
