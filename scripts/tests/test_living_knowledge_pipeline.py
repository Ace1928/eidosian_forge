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
