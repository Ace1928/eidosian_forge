#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _status_from_score(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def build_scorecard(repo_root: Path) -> dict[str, Any]:
    runtime_root = repo_root / "data" / "runtime"
    capabilities = _load_json(runtime_root / "platform_capabilities.json")
    scheduler = _load_json(runtime_root / "eidos_scheduler_status.json")
    coordinator = _load_json(runtime_root / "forge_coordinator_status.json")
    proof = _load_json(repo_root / "reports" / "proof" / "entity_proof_scorecard_latest.json")
    checks = {
        "platform_capabilities": (runtime_root / "platform_capabilities.json").exists(),
        "scheduler_status": bool(scheduler),
        "coordinator_status": bool(coordinator),
        "theory_of_operation": (repo_root / "docs" / "THEORY_OF_OPERATION.md").exists(),
        "termux_runit_installer": (repo_root / "scripts" / "install_termux_runit_services.sh").exists(),
        "boot_installer": (repo_root / "scripts" / "install_termux_boot.sh").exists(),
        "proof_scorecard": bool(proof),
        "scheduler_control": (repo_root / "scripts" / "eidos_scheduler_control.py").exists(),
    }
    sections = {
        "platform_portability": 0.0,
        "state_replayability": 0.0,
        "boot_and_services": 0.0,
        "operator_doctrine": 0.0,
    }
    if checks["platform_capabilities"]:
        sections["platform_portability"] += 0.5
    if capabilities.get("platform"):
        sections["platform_portability"] += 0.3
    if checks["proof_scorecard"]:
        sections["platform_portability"] += 0.1
    if checks["scheduler_status"]:
        sections["state_replayability"] += 0.35
    if checks["coordinator_status"]:
        sections["state_replayability"] += 0.35
    if checks["scheduler_control"]:
        sections["state_replayability"] += 0.15
    if checks["termux_runit_installer"]:
        sections["boot_and_services"] += 0.45
    if checks["boot_installer"]:
        sections["boot_and_services"] += 0.35
    if scheduler.get("state"):
        sections["boot_and_services"] += 0.1
    if checks["theory_of_operation"]:
        sections["operator_doctrine"] += 0.7
    if checks["proof_scorecard"]:
        sections["operator_doctrine"] += 0.2

    section_scores = {key: round(min(1.0, value), 6) for key, value in sections.items()}
    overall_score = round(sum(section_scores.values()) / max(1, len(section_scores)), 6)
    gaps = []
    if not checks["platform_capabilities"]:
        gaps.append("No platform capability artifact found.")
    if not checks["scheduler_status"]:
        gaps.append("No scheduler status artifact found.")
    if not checks["coordinator_status"]:
        gaps.append("No coordinator status artifact found.")
    if not checks["termux_runit_installer"]:
        gaps.append("No service supervision installer found.")
    if not checks["boot_installer"]:
        gaps.append("No boot installer found.")
    if not checks["theory_of_operation"]:
        gaps.append("No theory-of-operation document found.")
    return {
        "contract": "eidos.migration_replay_scorecard.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "checks": checks,
        "section_scores": section_scores,
        "overall_score": overall_score,
        "status": _status_from_score(overall_score),
        "gaps": gaps,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Migration Replay Scorecard",
        "",
        f"- Generated: `{payload.get('generated_at')}`",
        f"- Overall status: `{payload.get('status')}`",
        f"- Overall score: `{payload.get('overall_score')}`",
        "",
        "## Section Scores",
        "",
    ]
    for key, value in sorted((payload.get("section_scores") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gaps", ""])
    for gap in payload.get("gaps") or []:
        lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a migration/replay reproducibility scorecard.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else repo_root / "reports" / "proof"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = build_scorecard(repo_root)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = report_dir / f"migration_replay_scorecard_{stamp}.json"
    md_path = report_dir / f"migration_replay_scorecard_{stamp}.md"
    latest_json = report_dir / "migration_replay_scorecard_latest.json"
    latest_md = report_dir / "migration_replay_scorecard_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "latest": str(latest_json), "overall_score": payload["overall_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
