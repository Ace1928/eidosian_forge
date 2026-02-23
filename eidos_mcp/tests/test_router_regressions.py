from __future__ import annotations

import inspect
import json
from pathlib import Path

from eidos_mcp.routers import refactor as refactor_router
from eidos_mcp.routers import tiered_memory as tiered_router


def test_refactor_analyze_resolves_relative_paths_without_root_prefix(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "pkg" / "module.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    captured: dict[str, str] = {}

    class _FakeAnalyzer:
        def __init__(self, path: Path) -> None:
            captured["path"] = str(path)

        def analyze(self) -> dict:
            return {
                "file_info": {"path": captured["path"]},
                "modules": [{"name": "pkg.module"}],
                "dependencies": [],
            }

    monkeypatch.setattr(refactor_router, "CodeAnalyzer", _FakeAnalyzer)
    monkeypatch.setattr(refactor_router, "refactor", type("_RefactorModule", (), {"CodeAnalyzer": _FakeAnalyzer})())
    result = refactor_router.refactor_analyze("pkg/module.py")
    payload = json.loads(result)
    assert payload["file_info"]["path"] == str(source.resolve())
    assert captured["path"] == str(source.resolve())


def test_eidos_remember_self_no_importance_argument_regression(monkeypatch) -> None:
    class _FakeMemory:
        def remember_self(self, content: str, tags=None) -> str:
            return "mem_self_001"

    monkeypatch.setattr(tiered_router, "_get_tiered_memory", lambda: _FakeMemory())
    signature = inspect.signature(tiered_router.eidos_remember_self)
    assert "importance" not in signature.parameters
    result = tiered_router.eidos_remember_self(content="identity", tags=["test"])
    assert "Self-memory stored: mem_self_001" in result
