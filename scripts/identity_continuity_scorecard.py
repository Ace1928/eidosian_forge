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


def _load_any_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _status(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _continuity_metrics(core_bench: dict[str, Any], trial: dict[str, Any]) -> dict[str, float]:
    cap = core_bench.get("capability") if isinstance(core_bench.get("capability"), dict) else {}
    after = trial.get("after") if isinstance(trial.get("after"), dict) else {}
    phenom = after.get("phenomenology") if isinstance(after.get("phenomenology"), dict) else {}
    delta = trial.get("delta") if isinstance(trial.get("delta"), dict) else {}
    return {
        "agency": _safe_float(cap.get("agency")),
        "boundary_stability": _safe_float(cap.get("boundary_stability")),
        "continuity_index": _safe_float(
            cap.get("phenom_continuity_index"),
            _safe_float(phenom.get("continuity_index")),
        ),
        "ownership_index": _safe_float(
            cap.get("phenom_ownership_index"),
            _safe_float(phenom.get("ownership_index")),
        ),
        "perspective_coherence_index": _safe_float(
            cap.get("phenom_perspective_coherence_index"),
            _safe_float(phenom.get("perspective_coherence_index")),
        ),
        "continuity_delta": _safe_float(delta.get("continuity_delta")),
        "coherence_delta": _safe_float(delta.get("coherence_delta")),
    }


def build_scorecard(repo_root: Path) -> dict[str, Any]:
    reports_root = repo_root / "reports"
    runtime_root = repo_root / "data" / "runtime"
    proof_root = reports_root / "proof"

    core_bench = _load_json(reports_root / "consciousness_benchmarks" / "benchmark_latest.json")
    if not core_bench:
        for path in sorted((reports_root / "consciousness_benchmarks").glob("benchmark_*.json"), reverse=True):
            core_bench = _load_json(path)
            if core_bench:
                break
    trial = _load_json(reports_root / "consciousness_trials" / "trial_latest.json")
    if not trial:
        for path in sorted((reports_root / "consciousness_trials").glob("trial_*.json"), reverse=True):
            trial = _load_json(path)
            if trial:
                break

    session_context = _load_json(runtime_root / "session_bridge" / "latest_context.json")
    session_import = _load_json(runtime_root / "session_bridge" / "import_status.json")
    tiered_memory = _load_any_json(repo_root / "data" / "tiered_memory" / "working.json")
    proof = _load_json(proof_root / "entity_proof_scorecard_latest.json")

    theory_path = repo_root / "docs" / "THEORY_OF_OPERATION.md"
    manifesto_path = repo_root / "EIDOS_IDENTITY_MANIFESTO.md"
    ledger_path = runtime_root / "test_autonomy" / "ledger" / "continuity_ledger.jsonl"

    metrics = _continuity_metrics(core_bench, trial)
    recent_sessions = session_context.get("recent_sessions") if isinstance(session_context.get("recent_sessions"), list) else []
    imported_gemini = len((session_import.get("gemini") or {}).get("imported_ids") or [])
    imported_codex = len(((session_import.get("codex") or {}).get("threads") or {}).keys())
    memory_rows = tiered_memory if isinstance(tiered_memory, list) else []

    narrative_score = 0.0
    narrative_gaps: list[str] = []
    if theory_path.exists():
        narrative_score += 0.35
    else:
        narrative_gaps.append("No canonical theory-of-operation document found.")
    if manifesto_path.exists():
        narrative_score += 0.2
    else:
        narrative_gaps.append("No identity manifesto found.")
    if recent_sessions:
        narrative_score += 0.25
    else:
        narrative_gaps.append("No recent session bridge continuity context found.")
    if imported_codex or imported_gemini:
        narrative_score += 0.2
    else:
        narrative_gaps.append("No imported Codex/Gemini continuity state found.")

    memory_score = 0.0
    memory_gaps: list[str] = []
    if memory_rows:
        memory_score += 0.4
    else:
        memory_gaps.append("No tiered memory records detected in working memory.")
    if imported_codex:
        memory_score += 0.2
    else:
        memory_gaps.append("No Codex session continuity imported.")
    if imported_gemini:
        memory_score += 0.2
    else:
        memory_gaps.append("No Gemini session continuity imported.")
    if ledger_path.exists():
        memory_score += 0.2
    else:
        memory_gaps.append("No continuity ledger artifact detected.")

    agency_score = 0.0
    agency_gaps: list[str] = []
    if core_bench:
        agency_score += 0.25
    else:
        agency_gaps.append("No consciousness benchmark artifact found.")
    if trial:
        agency_score += 0.15
    else:
        agency_gaps.append("No perturbation trial artifact found.")
    if metrics["agency"] >= 0.9:
        agency_score += 0.2
    else:
        agency_gaps.append(f"Agency continuity is weak at `{metrics['agency']}`.")
    if metrics["boundary_stability"] >= 0.9:
        agency_score += 0.15
    else:
        agency_gaps.append(f"Boundary stability is weak at `{metrics['boundary_stability']}`.")
    if metrics["ownership_index"] >= 0.5:
        agency_score += 0.15
    else:
        agency_gaps.append(f"Ownership continuity is weak at `{metrics['ownership_index']}`.")
    if metrics["perspective_coherence_index"] >= 0.5:
        agency_score += 0.1
    else:
        agency_gaps.append(
            f"Perspective coherence continuity is weak at `{metrics['perspective_coherence_index']}`."
        )

    alignment_score = 0.0
    alignment_gaps: list[str] = []
    if proof:
        alignment_score += 0.25
    else:
        alignment_gaps.append("No latest proof scorecard found.")
    if metrics["continuity_index"] > 0.0:
        alignment_score += 0.25
    else:
        alignment_gaps.append("Phenomenological continuity remains zero or absent.")
    if metrics["continuity_delta"] >= -0.1:
        alignment_score += 0.2
    else:
        alignment_gaps.append(f"Continuity delta is too negative at `{metrics['continuity_delta']}`.")
    if theory_path.exists() and manifesto_path.exists():
        alignment_score += 0.15
    else:
        alignment_gaps.append("Theory and manifesto are not both present for identity doctrine alignment.")
    if recent_sessions and (imported_codex or imported_gemini):
        alignment_score += 0.15
    else:
        alignment_gaps.append("Cross-interface continuity imports are not strong enough yet.")

    sections = {
        "narrative_continuity": round(min(1.0, narrative_score), 6),
        "memory_continuity": round(min(1.0, memory_score), 6),
        "agency_continuity": round(min(1.0, agency_score), 6),
        "identity_alignment": round(min(1.0, alignment_score), 6),
    }
    overall_score = round(sum(sections.values()) / len(sections), 6)

    return {
        "contract": "eidos.identity_continuity_scorecard.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "repo_root": str(repo_root),
        "overall_score": overall_score,
        "status": _status(overall_score),
        "section_scores": sections,
        "continuity_metrics": metrics,
        "session_bridge": {
            "recent_sessions": len(recent_sessions),
            "imported_codex_threads": imported_codex,
            "imported_gemini_records": imported_gemini,
        },
        "memory": {
            "working_records": len(memory_rows),
            "continuity_ledger_present": ledger_path.exists(),
        },
        "identity_sources": {
            "theory_of_operation": theory_path.exists(),
            "identity_manifesto": manifesto_path.exists(),
            "proof_scorecard": bool(proof),
        },
        "gaps": narrative_gaps + memory_gaps + agency_gaps + alignment_gaps,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Identity Continuity Scorecard",
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
    lines.extend(["", "## Continuity Metrics", ""])
    for key, value in sorted((payload.get("continuity_metrics") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gaps", ""])
    for gap in payload.get("gaps") or []:
        lines.append(f"- {gap}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an identity continuity scorecard.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else repo_root / "reports" / "proof"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = build_scorecard(repo_root)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = report_dir / f"identity_continuity_scorecard_{stamp}.json"
    md_path = report_dir / f"identity_continuity_scorecard_{stamp}.md"
    latest_json = report_dir / "identity_continuity_scorecard_latest.json"
    latest_md = report_dir / "identity_continuity_scorecard_latest.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    latest_json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    latest_md.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({"json": str(json_path), "latest": str(latest_json), "overall_score": payload["overall_score"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
