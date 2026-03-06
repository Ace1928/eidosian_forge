from __future__ import annotations

import json
from pathlib import Path

from eidos_mcp.routers import word_forge


def test_wf_enrich_term_updates_graph(tmp_path: Path, monkeypatch) -> None:
    graph_path = tmp_path / "semantic_graph.json"
    queue_path = tmp_path / "queue.json"
    monkeypatch.setattr(word_forge, "SEMANTIC_GRAPH_PATH", graph_path)
    monkeypatch.setattr(word_forge, "LEXICON_QUEUE_PATH", queue_path)
    monkeypatch.setattr(word_forge, "_graph", None)
    monkeypatch.setattr(
        word_forge,
        "_generate_structured_payload",
        lambda **_: {
            "definition": "The capacity for self-directed action.",
            "pos": "noun",
            "phonetic": "aw-ton-uh-mee",
            "aliases": ["self-direction"],
            "examples": ["Autonomy improves resilience."],
            "roots": ["auto", "nomos"],
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
    assert nodes["autonomy"]["phonetic"] == "aw-ton-uh-mee"
    assert "agency" in nodes["autonomy"]["domains"]
    assert any(edge["source"] == "autonomy" and edge["target"] == "agency" for edge in saved["edges"])


def test_wf_build_lexicon_from_text_uses_model_payload(tmp_path: Path, monkeypatch) -> None:
    graph_path = tmp_path / "semantic_graph.json"
    queue_path = tmp_path / "queue.json"
    monkeypatch.setattr(word_forge, "SEMANTIC_GRAPH_PATH", graph_path)
    monkeypatch.setattr(word_forge, "LEXICON_QUEUE_PATH", queue_path)
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
                    "phonetic": "vek-tor",
                    "aliases": ["semantic vector"],
                    "examples": ["Vector search links related documents."],
                    "roots": ["vect"],
                    "domains": ["retrieval"],
                    "related_terms": [{"term": "embedding", "relation_type": "related", "weight": 0.8}],
                },
                {
                    "term": "embedding",
                    "definition": "A learned representation of text.",
                    "pos": "noun",
                    "phonetic": "em-bed-ding",
                    "aliases": [],
                    "examples": [],
                    "roots": ["embed"],
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


def test_wf_build_lexicon_from_text_falls_back_when_budget_denied(tmp_path: Path, monkeypatch) -> None:
    graph_path = tmp_path / "semantic_graph.json"
    queue_path = tmp_path / "queue.json"
    monkeypatch.setattr(word_forge, "SEMANTIC_GRAPH_PATH", graph_path)
    monkeypatch.setattr(word_forge, "LEXICON_QUEUE_PATH", queue_path)
    monkeypatch.setattr(word_forge, "_graph", None)
    monkeypatch.setattr(
        word_forge,
        "_generate_structured_payload",
        lambda **_: {
            "_budget_denied": True,
            "_budget_reason": "instance_budget_exceeded",
            "_effective_thinking_mode": "off",
        },
    )

    payload = json.loads(word_forge.wf_build_lexicon_from_text("Vector retrieval uses embeddings."))
    assert payload["status"] == "success"
    assert payload["budget_denied"] is True
    assert payload["budget_reason"] == "instance_budget_exceeded"


def test_wf_queue_and_process_lexicon_queue(tmp_path: Path, monkeypatch) -> None:
    graph_path = tmp_path / "semantic_graph.json"
    queue_path = tmp_path / "queue.json"
    monkeypatch.setattr(word_forge, "SEMANTIC_GRAPH_PATH", graph_path)
    monkeypatch.setattr(word_forge, "LEXICON_QUEUE_PATH", queue_path)
    monkeypatch.setattr(word_forge, "_graph", None)
    monkeypatch.setattr(
        word_forge,
        "_generate_structured_payload",
        lambda **_: {
            "definition": "A queued lexical term.",
            "pos": "noun",
            "phonetic": "vek-tor",
            "aliases": [],
            "examples": ["Vector search retrieves neighbors."],
            "roots": ["vect"],
            "domains": ["retrieval"],
            "related_terms": [],
            "_effective_thinking_mode": "on",
        },
    )

    queued = json.loads(word_forge.wf_queue_terms_from_text("Vector search improves retrieval.", source="unit:test"))
    processed = json.loads(word_forge.wf_process_lexicon_queue(max_terms=8))
    status = json.loads(word_forge.wf_lexicon_queue_status())

    assert queued["status"] == "success"
    assert queued["queue_size"] >= 1
    assert processed["enriched"] >= 1
    assert status["counts"]["enriched"] >= 1
