#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import time
from pathlib import Path
from typing import Any


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return str(dst)


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def export_bundle(repo_root: Path, output_root: Path) -> dict[str, Any]:
    stamp = _now_stamp()
    bundle_root = output_root / stamp
    bundle_root.mkdir(parents=True, exist_ok=True)

    proof_root = repo_root / "reports" / "proof"
    docs_root = repo_root / "docs"
    benchmarks_root = repo_root / "reports" / "external_benchmarks"

    files: list[dict[str, Any]] = []
    missing: list[str] = []

    def include(src: Path, relative_target: str, label: str) -> None:
        if not src.exists():
            missing.append(label)
            return
        target = bundle_root / relative_target
        _copy(src, target)
        files.append(
            {
                "label": label,
                "source": _rel(src, repo_root),
                "bundle_path": relative_target,
            }
        )

    include(proof_root / "entity_proof_scorecard_latest.json", "proof/entity_proof_scorecard_latest.json", "proof_json")
    include(proof_root / "entity_proof_scorecard_latest.md", "proof/entity_proof_scorecard_latest.md", "proof_markdown")
    include(
        proof_root / "migration_replay_scorecard_latest.json",
        "proof/migration_replay_scorecard_latest.json",
        "migration_json",
    )
    include(
        proof_root / "migration_replay_scorecard_latest.md",
        "proof/migration_replay_scorecard_latest.md",
        "migration_markdown",
    )
    include(
        proof_root / "identity_continuity_scorecard_latest.json",
        "proof/identity_continuity_scorecard_latest.json",
        "identity_continuity_json",
    )
    include(
        proof_root / "identity_continuity_scorecard_latest.md",
        "proof/identity_continuity_scorecard_latest.md",
        "identity_continuity_markdown",
    )
    recent_identity_history: list[dict[str, Any]] = []
    history_paths = [
        path
        for path in sorted(proof_root.glob("identity_continuity_scorecard_*.json"), reverse=True)
        if path.name != "identity_continuity_scorecard_latest.json"
    ]
    for path in history_paths[:5]:
        target = f"proof/identity_history/{path.name}"
        include(path, target, f"identity_history:{path.stem}")
        payload = _load_json(path)
        recent_identity_history.append(
            {
                "path": target,
                "generated_at": payload.get("generated_at"),
                "overall_score": payload.get("overall_score"),
                "status": payload.get("status"),
            }
        )
    include(docs_root / "THEORY_OF_OPERATION.md", "docs/THEORY_OF_OPERATION.md", "theory_of_operation")

    benchmark_rows: list[dict[str, Any]] = []
    if benchmarks_root.exists():
        for latest in sorted(benchmarks_root.glob("*/latest.json")):
            suite = latest.parent.name
            target = bundle_root / "external_benchmarks" / suite / "latest.json"
            _copy(latest, target)
            payload = _load_json(latest)
            row = {
                "suite": suite,
                "bundle_path": _rel(target, bundle_root),
                "source": _rel(latest, repo_root),
                "score": payload.get("score"),
                "status": payload.get("status"),
                "participant": payload.get("participant"),
                "execution_mode": payload.get("execution_mode"),
            }
            benchmark_rows.append(row)
            files.append(
                {
                    "label": f"external_benchmark:{suite}",
                    "source": _rel(latest, repo_root),
                    "bundle_path": _rel(target, bundle_root),
                }
            )
    else:
        missing.append("external_benchmarks_root")

    proof_payload = _load_json(proof_root / "entity_proof_scorecard_latest.json")
    migration_payload = _load_json(proof_root / "migration_replay_scorecard_latest.json")
    identity_payload = _load_json(proof_root / "identity_continuity_scorecard_latest.json")

    manifest = {
        "contract": "eidos.entity_proof_bundle.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": ".",
        "bundle_root": _rel(bundle_root, output_root),
        "proof_summary": proof_payload.get("overall", {}),
        "migration_summary": {
            "overall_score": migration_payload.get("overall_score"),
            "status": migration_payload.get("status"),
        },
        "identity_summary": {
            "overall_score": identity_payload.get("overall_score"),
            "status": identity_payload.get("status"),
            "history": identity_payload.get("history") if isinstance(identity_payload.get("history"), dict) else {},
            "recent_history": recent_identity_history,
        },
        "benchmarks": benchmark_rows,
        "files": files,
        "missing": sorted(set(missing)),
    }

    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    tar_path = output_root / f"entity_proof_bundle_{stamp}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as archive:
        archive.add(bundle_root, arcname=stamp)

    latest_manifest = output_root / "latest_manifest.json"
    latest_bundle = output_root / "latest_bundle.txt"
    latest_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    latest_bundle.write_text(_rel(tar_path, repo_root) + "\n", encoding="utf-8")

    return {
        "manifest": _rel(manifest_path, repo_root),
        "bundle": _rel(tar_path, repo_root),
        "latest_manifest": _rel(latest_manifest, repo_root),
        "latest_bundle": _rel(latest_bundle, repo_root),
        "proof_status": manifest["proof_summary"].get("status"),
        "proof_score": manifest["proof_summary"].get("score"),
        "benchmark_count": len(benchmark_rows),
        "missing": manifest["missing"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Export a publishable Eidos proof bundle.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output-root", default=None, help="Override output directory (default: reports/proof_bundle)")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else repo_root / "reports" / "proof_bundle"
    output_root.mkdir(parents=True, exist_ok=True)
    payload = export_bundle(repo_root, output_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
