from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "living_knowledge_pipeline.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader(
        "living_knowledge_pipeline",
        str(SCRIPT_PATH),
    )
    spec = importlib.util.spec_from_loader("living_knowledge_pipeline", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    loader.exec_module(module)
    return module


pipeline = _load_module()


def test_extract_memory_records_handles_mixed_payload(tmp_path: Path) -> None:
    payload = {
        "short_term": [{"id": "a", "content": "alpha"}, {"id": "b", "content": "beta"}],
        "episodic": {"id": "c", "content": "gamma"},
        "ignored": [{"id": "d", "content": ""}],
    }
    path = tmp_path / "memory_data.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    rows = pipeline.extract_memory_records(path)
    ids = {r["id"] for r in rows}
    assert "short_term:a" in ids
    assert "short_term:b" in ids
    assert "episodic:c" in ids
    assert all(r["content"] for r in rows)


def test_group_exact_duplicates() -> None:
    rec_a = pipeline.StagedRecord(
        doc_id="a",
        source_path="a.txt",
        kind="docs",
        sha256="hash-1",
        bytes=10,
        chars=10,
        staged_path="/tmp/a",
        simhash="0000000000000001",
    )
    rec_b = pipeline.StagedRecord(
        doc_id="b",
        source_path="b.txt",
        kind="docs",
        sha256="hash-1",
        bytes=10,
        chars=10,
        staged_path="/tmp/b",
        simhash="0000000000000001",
    )
    rec_c = pipeline.StagedRecord(
        doc_id="c",
        source_path="c.txt",
        kind="docs",
        sha256="hash-2",
        bytes=8,
        chars=8,
        staged_path="/tmp/c",
        simhash="0000000000000002",
    )
    groups = pipeline.group_exact_duplicates([rec_a, rec_b, rec_c])
    assert len(groups) == 1
    assert groups[0]["occurrences"] == 2
    assert set(groups[0]["documents"]) == {"a", "b"}


def test_detect_near_duplicates() -> None:
    text1 = "Kael protects the crystal while Seraphina supports Alaric."
    text2 = "Kael protects the crystal and Seraphina supports Alaric."
    sim1 = f"{pipeline._simhash64(text1):016x}"
    sim2 = f"{pipeline._simhash64(text2):016x}"
    rec_a = pipeline.StagedRecord("a", "a", "docs", "x1", 10, len(text1), "/tmp/a", sim1)
    rec_b = pipeline.StagedRecord("b", "b", "docs", "x2", 10, len(text2), "/tmp/b", sim2)
    pairs = pipeline.detect_near_duplicates([rec_a, rec_b], max_hamming=8)
    assert len(pairs) == 1
    assert {pairs[0]["doc_a"], pairs[0]["doc_b"]} == {"a", "b"}


def test_run_graphrag_index_uses_adapter_scan_roots(tmp_path: Path, monkeypatch) -> None:
    workspace = tmp_path / "workspace"
    docs = workspace / "input"
    docs.mkdir(parents=True, exist_ok=True)

    calls: list[tuple[str, list[Path]]] = []

    class FakeGraphRAG:
        def run_incremental_index(self, scan_roots):
            calls.append(("index", list(scan_roots)))
            return {
                "success": True,
                "mode": "native_vector_graph",
                "scan_roots": [str(p) for p in scan_roots],
                "community_reports": {"count": 1, "average_quality_score": 0.75},
                "report_trends": {"entries": 1, "latest": {"average_quality_score": 0.75}},
            }

    monkeypatch.setattr(pipeline, "_build_graphrag", lambda workspace_root: FakeGraphRAG())
    result = pipeline._run_graphrag_index(workspace, method="global", scan_roots=[docs])

    assert result["success"] is True
    assert result["community_reports"]["average_quality_score"] == 0.75
    assert result["report_trends"]["latest"]["average_quality_score"] == 0.75
    assert calls == [("index", [docs])]


def test_render_graphrag_query_result_handles_local_fallback() -> None:
    rendered = pipeline._render_graphrag_query_result(
        {
            "summary": "Top knowledge hit: unified graph",
            "knowledge_context": [{"content": "Knowledge node one"}],
            "memory_context": [{"content": "Memory node one"}],
            "graph_neighbors": [{"content": "Graph neighbor one"}],
        }
    )

    assert "Top knowledge hit: unified graph" in rendered
    assert "Knowledge node one" in rendered
    assert "Memory node one" in rendered
    assert "Graph neighbor one" in rendered


def test_generate_living_documentation_uses_central_qwen_config(tmp_path: Path, monkeypatch) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    captured: dict[str, object] = {}

    class FakeModelConfig:
        def generate_payload(self, prompt: str, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            return {
                "response": json.dumps(
                    {
                        "title": "Living Forge Summary",
                        "summary": "The forge is converging on a unified graph substrate.",
                        "key_findings": ["Native GraphRAG trends are available."],
                        "risks": ["Benchmark latency remains elevated."],
                        "priorities": ["Stabilize the documentation loop."],
                        "recommended_actions": ["Run a deeper quality sweep."],
                    }
                ),
                "thinking": "checked trend deltas",
            }

    monkeypatch.setattr(pipeline, "get_model_config", lambda: FakeModelConfig())

    result = pipeline.generate_living_documentation(
        run_root,
        repo_root=tmp_path,
        records_total=12,
        records_by_kind={"docs": 4, "code": 8},
        exact_duplicates=1,
        near_duplicates=2,
        drift={"added_count": 1, "removed_count": 0, "changed_count": 3},
        code_report={"run_stats": {"files_processed": 8, "units_created": 21}},
        graphrag_result={"indexed": True, "report_summary": {"count": 1}, "trend_summary": {"entries": 1}},
        config=pipeline.LivingDocumentationConfig(
            model="qwen3.5:2b",
            thinking_mode="on",
            timeout=900.0,
            max_tokens=1400,
            temperature=0.1,
        ),
    )

    assert result["generated"] is True
    assert result["model"] == "qwen3.5:2b"
    assert result["thinking_mode"] == "on"
    assert result["effective_thinking_mode"] == "on"
    assert result["fallback_used"] is False
    assert result["thinking_chars"] == len("checked trend deltas")
    assert (run_root / "living_documentation_summary.json").exists()
    assert (run_root / "living_documentation_summary.md").exists()
    assert captured["kwargs"]["model"] == "qwen3.5:2b"
    assert captured["kwargs"]["thinking_mode"] == "on"
    assert captured["kwargs"]["timeout"] == 900.0


def test_generate_living_documentation_retries_without_thinking_when_no_final_response(
    tmp_path: Path, monkeypatch
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir(parents=True, exist_ok=True)
    seen_modes: list[str] = []

    class FakeModelConfig:
        def generate_payload(self, prompt: str, **kwargs):
            mode = str(kwargs["thinking_mode"])
            seen_modes.append(mode)
            if mode == "on":
                return {"response": "", "thinking": "long hidden reasoning"}
            return {
                "response": json.dumps(
                    {
                        "title": "Recovered Summary",
                        "summary": "Fallback produced a usable final answer.",
                        "key_findings": ["The retry path worked."],
                        "risks": [],
                        "priorities": ["Keep thinking on as the first attempt."],
                        "recommended_actions": ["Retain fallback handling."],
                    }
                ),
                "thinking": "",
            }

    monkeypatch.setattr(pipeline, "get_model_config", lambda: FakeModelConfig())

    result = pipeline.generate_living_documentation(
        run_root,
        repo_root=tmp_path,
        records_total=4,
        records_by_kind={"docs": 4},
        exact_duplicates=0,
        near_duplicates=0,
        drift={"added_count": 0, "removed_count": 0, "changed_count": 1},
        code_report={"run_stats": {}},
        graphrag_result={"indexed": False},
        config=pipeline.LivingDocumentationConfig(model="qwen3.5:2b", thinking_mode="on"),
    )

    assert seen_modes == ["on", "off"]
    assert result["thinking_mode"] == "on"
    assert result["effective_thinking_mode"] == "off"
    assert result["fallback_used"] is True


def test_run_pipeline_includes_living_documentation_manifest(tmp_path: Path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    output_root = tmp_path / "output"
    workspace_root = tmp_path / "workspace"
    repo_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    workspace_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        pipeline,
        "run_code_analysis",
        lambda *args, **kwargs: {"run_stats": {}, "total_units": 0, "duplicate_group_count": 0},
    )
    monkeypatch.setattr(pipeline, "stage_repo_text_documents", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "stage_memory_and_kb_documents", lambda *args, **kwargs: [])
    monkeypatch.setattr(pipeline, "group_exact_duplicates", lambda records: [])
    monkeypatch.setattr(pipeline, "detect_near_duplicates", lambda records: [])
    monkeypatch.setattr(
        pipeline,
        "compare_with_previous_run",
        lambda *args, **kwargs: {"added_count": 0, "removed_count": 0, "changed_count": 0},
    )
    monkeypatch.setattr(
        pipeline,
        "generate_living_documentation",
        lambda *args, **kwargs: {
            "enabled": True,
            "generated": True,
            "model": "qwen3.5:2b",
            "thinking_mode": "off",
            "effective_thinking_mode": "off",
            "fallback_used": False,
            "json_path": "x.json",
            "markdown_path": "x.md",
        },
    )

    manifest = pipeline.run_pipeline(
        repo_root=repo_root,
        output_root=output_root,
        workspace_root=workspace_root,
        max_file_bytes=1000,
        max_chars_per_doc=2000,
        code_max_files=None,
        run_graphrag=False,
        queries=[],
        method="fast",
        living_doc_config=pipeline.LivingDocumentationConfig(
            model="qwen3.5:2b",
            thinking_mode="off",
            timeout=900.0,
            max_tokens=1200,
            temperature=0.1,
        ),
    )

    assert manifest["living_documentation"]["generated"] is True
    assert manifest["living_documentation"]["model"] == "qwen3.5:2b"
    assert manifest["living_documentation"]["thinking_mode"] == "off"
