from __future__ import annotations

import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from eidos_mcp.routers import code as code_router
from eidos_mcp.routers import glyph as glyph_router
from eidos_mcp.routers import knowledge as knowledge_router
from eidos_mcp.routers import refactor as refactor_router
from eidos_mcp.routers import repo as repo_router
from eidos_mcp.routers import system as system_router
from eidos_mcp.routers import tika as tika_router
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
        def remember_self(self, content: str, subdomain: str = "autobiography", tags=None) -> str:
            return "mem_self_001"

    monkeypatch.setattr(tiered_router, "_get_tiered_memory", lambda: _FakeMemory())
    signature = inspect.signature(tiered_router.eidos_remember_self)
    assert "importance" not in signature.parameters
    result = tiered_router.eidos_remember_self(content="identity", tags=["test"])
    assert "Self-memory stored: mem_self_001" in result


def test_eidos_remember_lesson_accepts_comma_delimited_tags(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeMemory:
        def remember_lesson(self, lesson: str, context: str | None = None, tags=None) -> str:
            captured["lesson"] = lesson
            captured["context"] = context
            captured["tags"] = tags
            return "lesson_001"

    monkeypatch.setattr(tiered_router, "_get_tiered_memory", lambda: _FakeMemory())
    result = tiered_router.eidos_remember_lesson("keep traces", context="tests", tags="proof, continuity , atlas")
    assert "Lesson stored: lesson_001" in result
    assert captured["tags"] == {"proof", "continuity", "atlas"}


def test_tika_ingest_directory_accepts_string_extensions_and_tags(monkeypatch, tmp_path: Path) -> None:
    source_dir = tmp_path / "docs"
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "notes.md").write_text("# Notes\n", encoding="utf-8")

    captured: list[dict[str, object]] = []

    class _FakeIngester:
        def ingest_file(self, file_path: Path, tags=None, chunk_size: int = 2000) -> dict:
            captured.append({"file_path": str(file_path), "tags": tags, "chunk_size": chunk_size})
            return {"status": "success", "nodes_created": 2}

    monkeypatch.setattr(tika_router, "_get_ingester", lambda: _FakeIngester())
    payload = json.loads(
        tika_router.tika_ingest_directory(
            str(source_dir),
            extensions="md,txt",
            tags="research, proof",
            recursive=False,
        )
    )
    assert payload["files_found"] == 1
    assert payload["files_processed"] == 1
    assert payload["nodes_created"] == 2
    assert captured[0]["tags"] == ["research", "proof"]


def test_kb_add_fact_accepts_comma_delimited_tags(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeKB:
        persistence_path = tmp_path / "kb.json"

        def add_knowledge(self, fact: str, tags=None):
            captured["fact"] = fact
            captured["tags"] = tags
            return SimpleNamespace(id="node_001")

    monkeypatch.setattr(knowledge_router, "kb", _FakeKB())
    monkeypatch.setattr(knowledge_router, "begin_transaction", lambda *args, **kwargs: nullcontext(SimpleNamespace(id="txn1")))
    result = knowledge_router.kb_add_fact("forge fact", "proof, atlas")
    assert "Added node: node_001" in result
    assert captured["tags"] == ["proof", "atlas"]


def test_grag_index_accepts_comma_delimited_scan_roots(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class _FakeGrag:
        def run_incremental_index(self, root_paths):
            captured["roots"] = [str(path) for path in root_paths]
            return {"ok": True}

    monkeypatch.setattr(knowledge_router, "grag", _FakeGrag())
    result = knowledge_router.grag_index(f"{tmp_path}/one,{tmp_path}/two")
    assert "ok" in result
    assert captured["roots"] == [str(tmp_path / "one"), str(tmp_path / "two")]


def test_code_search_accepts_string_element_types(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Elem:
        element_type = "function"
        name = "run"
        qualified_name = "pkg.run"
        file_path = "pkg.py"
        docstring = ""
        args = []
        methods = []

    class _FakeIndexer:
        def search(self, query: str, element_types=None):
            captured["query"] = query
            captured["element_types"] = element_types
            return [_Elem()]

    monkeypatch.setattr(code_router, "_get_indexer", lambda: _FakeIndexer())
    payload = json.loads(code_router.code_search("runner", element_types="function,class", limit=1))
    assert payload[0]["qualified_name"] == "pkg.run"
    assert captured["element_types"] == ["function", "class"]


def test_run_shell_command_accepts_string_transaction_paths(monkeypatch, tmp_path: Path) -> None:
    target = tmp_path / "data.txt"
    target.write_text("x", encoding="utf-8")

    class _Txn:
        id = "txn_cmd"

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(system_router, "_resolve_path", lambda raw: Path(raw))
    monkeypatch.setattr(system_router, "_is_allowed", lambda path: True)
    monkeypatch.setattr(system_router, "begin_transaction", lambda *args, **kwargs: _Txn())
    monkeypatch.setattr(system_router, "_is_read_only_command", lambda command: False)
    monkeypatch.setattr(
        system_router.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="ok\n", stderr="", returncode=0),
    )
    payload = json.loads(
        system_router.run_shell_command(
            "echo ok",
            safe_mode=True,
            transaction_paths=f"{target}",
        )
    )
    assert payload["exit_code"] == 0


def test_repo_create_docs_accepts_string_languages(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    def _fake_docs(base_path: Path, languages=None, overwrite: bool = True):
        captured["base_path"] = str(base_path)
        captured["languages"] = languages
        captured["overwrite"] = overwrite
        return {"ok": True}

    monkeypatch.setattr(repo_router.docs, "create_documentation_structure", _fake_docs)
    payload = repo_router.repo_create_docs(base_path=str(tmp_path), languages="python,rust", overwrite=False)
    assert payload["ok"] is True
    assert captured["languages"] == ["python", "rust"]


def test_glyph_text_to_banner_accepts_string_effects(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_banner(**kwargs):
        captured.update(kwargs)
        return "banner"

    monkeypatch.setattr(glyph_router, "text_to_banner", _fake_banner)
    result = glyph_router.glyph_text_to_banner("Eidos", effects="shadow, outline")
    assert result == "banner"
    assert captured["effects"] == ["shadow", "outline"]
