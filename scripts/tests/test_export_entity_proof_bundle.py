from __future__ import annotations

import importlib.util
import json
import tarfile
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[1] / "export_entity_proof_bundle.py"
_SPEC = importlib.util.spec_from_file_location("export_entity_proof_bundle", _MODULE_PATH)
assert _SPEC and _SPEC.loader
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
export_bundle = _MODULE.export_bundle


def _write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")


def test_export_bundle_collects_latest_proof_and_benchmarks(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(
        repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json",
        json.dumps({"overall": {"status": "yellow", "score": 0.7}}),
    )
    _write(repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.md", "# proof\n")
    _write(
        repo_root / "reports" / "proof" / "migration_replay_scorecard_latest.json",
        json.dumps({"overall_score": 0.8, "status": "green"}),
    )
    _write(repo_root / "reports" / "proof" / "migration_replay_scorecard_latest.md", "# migration\n")
    _write(
        repo_root / "reports" / "proof" / "identity_continuity_scorecard_latest.json",
        json.dumps({"overall_score": 0.77, "status": "yellow"}),
    )
    _write(repo_root / "reports" / "proof" / "identity_continuity_scorecard_latest.md", "# identity\n")
    _write(repo_root / "docs" / "THEORY_OF_OPERATION.md", "# theory\n")
    _write(
        repo_root / "reports" / "external_benchmarks" / "agencybench" / "latest.json",
        json.dumps({"suite": "agencybench", "score": 1.0, "status": "green"}),
    )

    result = export_bundle(repo_root, repo_root / "reports" / "proof_bundle")

    manifest_path = repo_root / result["manifest"]
    bundle_path = repo_root / result["bundle"]
    assert manifest_path.exists()
    assert bundle_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["proof_summary"]["status"] == "yellow"
    assert manifest["proof_summary"]["score"] == 0.7
    assert manifest["migration_summary"]["status"] == "green"
    assert manifest["benchmarks"][0]["suite"] == "agencybench"
    assert manifest["missing"] == []
    assert any(item["label"] == "identity_continuity_json" for item in manifest["files"])
    with tarfile.open(bundle_path, "r:gz") as archive:
        names = archive.getnames()
    assert any(name.endswith("manifest.json") for name in names)
    assert any(name.endswith("external_benchmarks/agencybench/latest.json") for name in names)


def test_export_bundle_reports_missing_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _write(
        repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json",
        json.dumps({"overall": {"status": "red", "score": 0.2}}),
    )

    result = export_bundle(repo_root, repo_root / "reports" / "proof_bundle")
    manifest = json.loads((repo_root / result["manifest"]).read_text(encoding="utf-8"))
    assert "migration_json" in manifest["missing"]
    assert "theory_of_operation" in manifest["missing"]
