from __future__ import annotations

import pytest

from knowledge_forge import KnowledgeForge

pytest.importorskip("pyvis", reason="pyvis required for visualization tests")


def test_visualize_pyvis_exports_html(tmp_path):
    kb = KnowledgeForge(persistence_path=tmp_path / "kb.json")
    a = kb.add_knowledge("A node", concepts=["alpha"], tags=["group-a"])
    b = kb.add_knowledge("B node", concepts=["beta"], tags=["group-b"])
    kb.link_nodes(a.id, b.id)

    out = tmp_path / "viz.html"
    report = kb.visualize_pyvis(out, max_nodes=10)
    assert report["node_count"] == 2
    assert report["edge_count"] == 1
    assert out.exists()
    text = out.read_text(encoding="utf-8", errors="ignore")
    assert "<html" in text.lower()
