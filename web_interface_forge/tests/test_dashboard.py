from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from web_interface_forge.dashboard import main as dashboard


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_doc_status_api_and_index_page(monkeypatch, tmp_path: Path) -> None:
    runtime = tmp_path / "doc_forge" / "runtime"
    final_docs = runtime / "final_docs"
    data_runtime = tmp_path / "data" / "runtime"
    final_docs.mkdir(parents=True, exist_ok=True)
    data_runtime.mkdir(parents=True, exist_ok=True)

    _write_json(
        runtime / "processor_status.json",
        {
            "status": "running",
            "processed": 12,
            "remaining": 4,
            "average_quality_score": 0.88,
            "last_approved": "foo/bar.py",
        },
    )
    _write_json(
        runtime / "doc_index.json",
        {
            "entries": [
                {
                    "source": "foo/bar.py",
                    "document": "foo/bar.py.md",
                    "score": 0.91,
                    "doc_type": "py",
                    "updated_at": "2026-02-26T00:00:00+00:00",
                }
            ]
        },
    )

    (final_docs / "foo").mkdir(parents=True, exist_ok=True)
    (final_docs / "foo" / "bar.py.md").write_text("# Example\n", encoding="utf-8")

    monkeypatch.setattr(dashboard, "DOC_RUNTIME", runtime)
    monkeypatch.setattr(dashboard, "DOC_FINAL", final_docs)
    monkeypatch.setattr(dashboard, "DOC_INDEX", runtime / "doc_index.json")
    monkeypatch.setattr(dashboard, "DOC_STATUS", runtime / "processor_status.json")
    monkeypatch.setattr(dashboard, "PIPELINE_STATUS", data_runtime / "living_pipeline_status.json")
    monkeypatch.setattr(dashboard, "SCHEDULER_STATUS", data_runtime / "eidos_scheduler_status.json")
    monkeypatch.setattr(
        dashboard,
        "get_forge_overview",
        lambda: {
            "system": {"cpu": 2.5, "ram_percent": 12.0},
            "forge_status": {"doc_forge": "running", "pipeline": "running", "scheduler": "idle"},
            "documents": dashboard.get_doc_snapshot(),
            "pipeline": {
                "pipeline": {"phase": "word_forge", "eta_seconds": 42, "records_total": 8},
                "scheduler": {"state": "idle", "cycle": 3, "consecutive_failures": 0},
            },
            "word_forge": {"term_count": 2, "edge_count": 1, "sample_terms": [{"term": "vector"}]},
            "code_forge": {"total_units": 4},
            "knowledge": {
                "node_count": 3,
                "assessment_summary": {"status": "stable", "score": 0.72, "weak_community_labels": []},
                "report_summary": {"top_community": "documents"},
            },
        },
    )
    monkeypatch.setattr(
        dashboard,
        "search_knowledge_graph",
        lambda query, limit=12: {"query": query, "count": 1, "results": [{"id": "kb-1", "content": "Graph hit"}]},
    )
    monkeypatch.setattr(
        dashboard,
        "search_word_forge",
        lambda query, limit=12: {"query": query, "count": 1, "terms": [{"id": "vector", "definition": "embedding"}]},
    )
    monkeypatch.setattr(
        dashboard,
        "search_code_library",
        lambda query, limit=12: {
            "query": query,
            "count": 1,
            "results": [
                {
                    "id": "u1",
                    "qualified_name": "pkg.mod.fn",
                    "unit_type": "function",
                    "file_path": "src/pkg/mod.py",
                    "search_preview": "def fn(): pass",
                }
            ],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "get_code_unit_context",
        lambda unit_id: {"found": True, "unit": {"id": unit_id, "qualified_name": "pkg.mod.fn"}},
    )
    monkeypatch.setattr(
        dashboard,
        "get_code_graph",
        lambda limit_edges=300: {"available": True, "nodes": [{"id": "a.py"}], "edges": [], "summary": {"node_count": 1}},
    )
    monkeypatch.setattr(
        dashboard,
        "get_runtime_coordinator",
        lambda: {
            "available": True,
            "state": "running",
            "task": "living_documentation",
            "active_models": [{"model": "qwen3.5:2b"}],
            "history": [{"task": "word_forge", "state": "running", "active_model_count": 1}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "get_runtime_history",
        lambda limit=24: {
            "count": 1,
            "history": [{"task": "word_forge", "state": "running", "active_model_count": 1}],
            "current": {},
        },
    )
    monkeypatch.setattr(
        dashboard,
        "get_memory_snapshot",
        lambda: {
            "count": 3,
            "tiers": {"working": 1, "long_term": 2},
            "namespaces": {"knowledge": 2, "task": 1},
            "community_count": 1,
            "top_communities": [{"community": "knowledge:runtime:qwen", "count": 2}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "search_memory",
        lambda query, limit=12: {
            "query": query,
            "count": 1,
            "results": [{"id": "mem-1", "content": "memory hit", "tier": "working", "community": "knowledge:runtime:qwen"}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "search_docs",
        lambda query, limit=12: {
            "query": query,
            "count": 1,
            "results": [{"source": "foo/bar.py", "document": "foo/bar.py.md", "doc_type": "py"}],
        },
    )
    monkeypatch.setattr(
        dashboard,
        "get_unified_graph",
        lambda limit_per_domain=80, limit_edges=320: {
            "available": True,
            "nodes": [{"id": "knowledge:k1", "domain": "knowledge", "label": "Graph hit", "ref_id": "k1"}],
            "edges": [],
            "summary": {"node_count": 1, "edge_count": 0},
        },
    )
    monkeypatch.setattr(
        dashboard,
        "get_memory_graph",
        lambda limit=120: {"available": True, "nodes": [{"id": "mem-1"}], "edges": [], "summary": {"node_count": 1, "edge_count": 0, "community_count": 1}},
    )
    monkeypatch.setattr(
        dashboard,
        "get_explorer_node",
        lambda domain, node_id: {"found": True, "domain": domain, "node": {"id": node_id, "content": "detail"}},
    )
    monkeypatch.setattr(
        dashboard,
        "get_graph_neighbors",
        lambda domain, node_id, limit=20, depth=1: {
            "found": True,
            "focus_id": f"{domain}:{node_id}",
            "nodes": [{"id": f"{domain}:{node_id}", "domain": domain}],
            "edges": [],
            "summary": {"node_count": 1, "edge_count": 0, "depth": depth},
        },
    )

    with TestClient(dashboard.app) as client:
        resp = client.get("/api/doc/status")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["index_count"] == 1
        assert payload["status"]["processed"] == 12

        html = client.get("/").text
        assert "foo/bar.py" in html
        assert "Knowledge Search" in html
        explore_html = client.get("/explore").text
        assert "Atlas Explorer" in explore_html
        assert "Unified Query" in explore_html

        runtime_payload = client.get("/api/runtime/forge").json()
        assert runtime_payload["pipeline"]["pipeline"]["phase"] == "word_forge"
        coordinator_payload = client.get("/api/runtime/coordinator").json()
        assert coordinator_payload["task"] == "living_documentation"
        history_payload = client.get("/api/runtime/history").json()
        assert history_payload["count"] >= 1

        graph_payload = client.get("/api/graph/search?query=graph").json()
        assert graph_payload["count"] == 1
        unified_graph_payload = client.get("/api/graph/unified").json()
        assert unified_graph_payload["summary"]["node_count"] == 1

        lexicon_payload = client.get("/api/lexicon/search?query=vector").json()
        assert lexicon_payload["terms"][0]["id"] == "vector"

        code_payload = client.get("/api/code/search?query=fn").json()
        assert code_payload["results"][0]["qualified_name"] == "pkg.mod.fn"

        unit_payload = client.get("/api/code/unit/u1").json()
        assert unit_payload["unit"]["id"] == "u1"
        memory_payload = client.get("/api/memory/search?query=memory").json()
        assert memory_payload["count"] == 1
        memory_community_payload = client.get("/api/memory/communities").json()
        assert memory_community_payload["community_count"] == 1
        docs_payload = client.get("/api/docs/search?query=foo").json()
        assert docs_payload["results"][0]["source"] == "foo/bar.py"
        memory_graph_payload = client.get("/api/graph/memory").json()
        assert memory_graph_payload["summary"]["community_count"] == 1
        explorer_payload = client.get("/api/explorer/search?query=graph").json()
        assert explorer_payload["memory"]["count"] == 1
        node_payload = client.get("/api/explorer/node/knowledge/k1").json()
        assert node_payload["node"]["id"] == "k1"
        neighbor_payload = client.get("/api/explorer/neighbors/knowledge/k1").json()
        assert neighbor_payload["summary"]["node_count"] == 1
        assert neighbor_payload["summary"]["depth"] == 1


def test_browse_blocks_path_traversal() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.get("/browse/%2e%2e/%2e%2e/etc/passwd")
        assert resp.status_code == 403


def test_health_endpoint() -> None:
    with TestClient(dashboard.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["status"] == "ok"
        assert payload["service"] == "eidos_atlas"


def test_memory_rows_support_list_based_tier_files(monkeypatch, tmp_path: Path) -> None:
    memory_dir = tmp_path / "tiered_memory"
    _write_json(
        memory_dir / "long_term.json",
        [
            {
                "id": "mem-1",
                "content": "Vector-native memory record",
                "tier": "long_term",
                "namespace": "knowledge",
                "tags": ["vector", "memory"],
                "metadata": {"community": "knowledge:memory:vector"},
            }
        ],
    )
    monkeypatch.setattr(dashboard, "MEMORY_DIR", memory_dir)
    rows = dashboard._memory_rows()
    snapshot = dashboard.get_memory_snapshot()
    graph = dashboard.get_memory_graph(limit=10)

    assert rows[0]["community"] == "knowledge:memory:vector"
    assert snapshot["community_count"] == 1
    assert graph["summary"]["node_count"] == 1
