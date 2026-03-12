from __future__ import annotations

from eidos_mcp.routers import consciousness


class _Runner:
    def status(self):
        return {
            "timestamp": "2026-03-13T00:00:00Z",
            "state_dir": "/tmp/state",
            "memory_bridge": {
                "available": None,
                "introspector_available": None,
                "query": None,
                "recall_count": None,
                "last_error": None,
            },
            "knowledge_bridge": {
                "available": None,
                "query": None,
                "total_hits": None,
                "last_error": None,
            },
            "memory_recalls": None,
            "knowledge_hits": None,
        }


def test_bridge_status_payload_falls_back_to_direct_probes(monkeypatch) -> None:
    monkeypatch.setattr(consciousness, "_runner", lambda state_dir=None: _Runner())
    monkeypatch.setattr(
        consciousness,
        "_direct_memory_bridge_status",
        lambda: {
            "available": True,
            "introspector_available": True,
            "query": "",
            "recall_count": 0,
            "last_error": "",
            "stats": {"total": 8},
            "probe_source": "direct",
        },
    )
    monkeypatch.setattr(
        consciousness,
        "_direct_knowledge_bridge_status",
        lambda: {
            "available": True,
            "query": "",
            "total_hits": 0,
            "last_error": "",
            "stats": {"knowledge_count": 5},
            "probe_source": "direct",
        },
    )
    payload = consciousness._bridge_status_payload()
    assert payload["memory_bridge"]["available"] is True
    assert payload["memory_bridge"]["probe_source"] == "direct"
    assert payload["knowledge_bridge"]["available"] is True
    assert payload["knowledge_hits"] == 0
