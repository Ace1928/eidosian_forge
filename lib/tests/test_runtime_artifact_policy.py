from __future__ import annotations

import json
import subprocess
from pathlib import Path

from eidosian_runtime.artifact_policy import audit_runtime_artifacts, write_runtime_artifact_audit


def _commit_all(repo: Path) -> None:
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test User"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "init"], check=True)


def test_audit_runtime_artifacts_flags_tracked_generated_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True)

    (repo / "cfg").mkdir()
    (repo / "cfg" / "runtime_artifact_policy.json").write_text(
        json.dumps(
            {
                "tracked_generated_globs": [
                    "data/**/vectors/index.bin",
                    "data/runtime/**/*.jsonl",
                ],
                "live_generated_globs": [
                    "data/**/vectors/index.bin",
                    "data/runtime/**/*.jsonl",
                    "data/**/*.tmp",
                ],
            }
        ),
        encoding="utf-8",
    )
    (repo / "data" / "tiered_memory" / "vectors").mkdir(parents=True)
    (repo / "data" / "tiered_memory" / "vectors" / "index.bin").write_bytes(b"bin")
    (repo / "data" / "runtime" / "local_mcp_agent").mkdir(parents=True)
    (repo / "data" / "runtime" / "local_mcp_agent" / "history.jsonl").write_text("{}\n", encoding="utf-8")
    (repo / "src").mkdir()
    (repo / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    _commit_all(repo)

    (repo / "data" / ".kb.json.tmp").write_text("tmp\n", encoding="utf-8")

    report = audit_runtime_artifacts(repo)

    assert report["tracked_violation_count"] == 2
    assert "data/tiered_memory/vectors/index.bin" in report["tracked_generated_files"]
    assert "data/runtime/local_mcp_agent/history.jsonl" in report["tracked_generated_files"]
    assert "data/.kb.json.tmp" in report["live_generated_files"]


def test_write_runtime_artifact_audit_persists_report(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True)

    (repo / "cfg").mkdir()
    (repo / "cfg" / "runtime_artifact_policy.json").write_text(
        json.dumps(
            {
                "tracked_generated_globs": ["data/runtime/*_status.json"],
                "live_generated_globs": ["data/runtime/*_status.json"],
            }
        ),
        encoding="utf-8",
    )
    (repo / "data" / "runtime").mkdir(parents=True)
    (repo / "data" / "runtime" / "entity_proof_status.json").write_text("{}\n", encoding="utf-8")
    _commit_all(repo)

    output = repo / "reports" / "runtime_artifact_audit.json"
    report = write_runtime_artifact_audit(repo, output)

    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert persisted["tracked_violation_count"] == 1
    assert persisted == report


def test_audit_runtime_artifacts_collapses_directory_wide_runtime_roots(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "-C", str(repo), "init"], check=True)

    (repo / "cfg").mkdir()
    (repo / "cfg" / "runtime_artifact_policy.json").write_text(
        json.dumps(
            {
                "tracked_generated_globs": [],
                "live_generated_globs": ["data/runtime/external_benchmarks/**"],
            }
        ),
        encoding="utf-8",
    )
    (repo / "data" / "runtime" / "external_benchmarks" / "agencybench" / "run1").mkdir(parents=True)
    (repo / "data" / "runtime" / "external_benchmarks" / "agencybench" / "run1" / "status.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    _commit_all(repo)

    report = audit_runtime_artifacts(repo)

    assert report["live_generated_files"] == ["data/runtime/external_benchmarks"]
