from __future__ import annotations

from pathlib import Path

from eidos_mcp.routers import desire as desire_router


def test_consciousness_explain_desire_formats_internal_state(monkeypatch) -> None:
    class _StateStore:
        @staticmethod
        def get_module_state(name: str):
            mapping = {
                "affect": {"modulators": {"ambition": 0.9, "pride": 0.8, "satisfaction": 0.6}},
                "intero": {"drives": {"curiosity": 0.85, "threat": 0.2}},
                "phenomenology_probe": {"unity_index": 0.7, "ownership_index": 0.65},
            }
            return mapping.get(name, {})

    class _Kernel:
        def __init__(self, *_args, **_kwargs):
            self.state_store = _StateStore()

    monkeypatch.setattr(desire_router, "ConsciousnessKernel", _Kernel)

    result = desire_router.consciousness_explain_desire()

    assert "Eidosian Desire Proof" in result
    assert "Systemic Exploration" in result
    assert "Forge Expansion" in result
    assert "Self-Coherence Preservation" in result
    assert "Unity Index" in result
    assert "State Dir" in result


def test_consciousness_explain_desire_handles_kernel_failure(monkeypatch) -> None:
    class _BrokenKernel:
        def __init__(self, *_args, **_kwargs):
            raise RuntimeError("kernel unavailable")

    monkeypatch.setattr(desire_router, "ConsciousnessKernel", _BrokenKernel)

    result = desire_router.consciousness_explain_desire()

    assert "Error analyzing internal state" in result


def test_consciousness_explain_desire_uses_configured_state_dir(monkeypatch) -> None:
    captured: dict[str, Path] = {}

    class _StateStore:
        @staticmethod
        def get_module_state(name: str):
            mapping = {
                "affect": {"modulators": {"ambition": 0.1, "pride": 0.2}},
                "intero": {"drives": {"curiosity": 0.2}},
                "phenomenology_probe": {"unity_index": 0.4, "ownership_index": 0.4},
            }
            return mapping.get(name, {})

    class _Kernel:
        def __init__(self, state_dir, *_args, **_kwargs):
            captured["state_dir"] = Path(state_dir)
            self.state_store = _StateStore()

    monkeypatch.setenv("EIDOS_CONSCIOUSNESS_STATE_DIR", "/tmp/eidos-state")
    monkeypatch.setattr(desire_router, "ConsciousnessKernel", _Kernel)

    result = desire_router.consciousness_explain_desire()

    assert captured["state_dir"] == Path("/tmp/eidos-state")
    assert "Maintain adaptive equilibrium" in result
