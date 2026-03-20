#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception:
        return None
    if raw.startswith("version https://git-lfs.github.com/spec/v1"):
        return {
            "contract": "eidos.lfs_pointer.v1",
            "pointer": True,
            "path": str(path),
        }
    try:
        data = json.loads(raw)
    except Exception:
        return None
    if isinstance(data, dict):
        return data
    return None


def _latest_existing(paths: Iterable[Path]) -> Path | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda item: item.stat().st_mtime_ns)


def _latest_json(paths: Iterable[Path]) -> tuple[Path | None, dict[str, Any] | None]:
    newest_pointer: tuple[Path | None, dict[str, Any] | None] = (None, None)
    for path in sorted(
        (item for item in paths if item.exists()), key=lambda item: item.stat().st_mtime_ns, reverse=True
    ):
        payload = _load_json(path)
        if payload is None:
            continue
        if _has_unresolved_pointer(payload):
            if newest_pointer[0] is None:
                newest_pointer = (path, payload)
            continue
        return path, payload
    return newest_pointer


def _latest_glob(root: Path, pattern: str) -> tuple[Path | None, dict[str, Any] | None]:
    return _latest_json(sorted(root.glob(pattern)))


def _latest_named_json(root: Path, latest_name: str, pattern: str) -> tuple[Path | None, dict[str, Any] | None]:
    latest = root / latest_name
    payload = _load_json(latest)
    if payload:
        return latest, payload
    return _latest_glob(root, pattern)


def _path_age_days(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    age_s = max(0.0, time.time() - path.stat().st_mtime)
    return round(age_s / 86400.0, 3)


def _artifact_state(path: Path | None, *, window_days: int) -> dict[str, Any]:
    age_days = _path_age_days(path)
    exists = bool(path and path.exists())
    fresh = bool(exists and age_days is not None and age_days <= max(1, int(window_days)))
    stale = bool(exists and age_days is not None and age_days > max(1, int(window_days)))
    return {
        "path": str(path) if path else None,
        "exists": exists,
        "age_days": age_days,
        "fresh": fresh,
        "stale": stale,
    }


def _git_head(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _git_dirty(repo_root: Path) -> bool | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception:
        return None
    return bool(result.stdout.strip())


def _status_from_score(score: float) -> str:
    if score >= 0.8:
        return "green"
    if score >= 0.5:
        return "yellow"
    return "red"


def _category(
    name: str,
    score: float,
    strengths: list[str],
    gaps: list[str],
    evidence_paths: list[str],
) -> dict[str, Any]:
    return {
        "category": name,
        "score": round(float(score), 6),
        "status": _status_from_score(score),
        "strengths": strengths,
        "gaps": gaps,
        "evidence_paths": sorted({path for path in evidence_paths if path}),
    }


def _continuity_summary(bench: dict[str, Any] | None, trial: dict[str, Any] | None) -> dict[str, Any]:
    capability = (bench or {}).get("capability") if isinstance((bench or {}).get("capability"), dict) else {}
    before = (trial or {}).get("before") if isinstance((trial or {}).get("before"), dict) else {}
    after = (trial or {}).get("after") if isinstance((trial or {}).get("after"), dict) else {}
    before_phenom = before.get("phenomenology") if isinstance(before.get("phenomenology"), dict) else {}
    after_phenom = after.get("phenomenology") if isinstance(after.get("phenomenology"), dict) else {}
    return {
        "coherence_ratio": _safe_float(capability.get("coherence_ratio")),
        "agency": _safe_float(capability.get("agency")),
        "boundary_stability": _safe_float(capability.get("boundary_stability")),
        "continuity_index": _safe_float(
            capability.get("phenom_continuity_index"),
            _safe_float(after_phenom.get("continuity_index")),
        ),
        "ownership_index": _safe_float(
            capability.get("phenom_ownership_index"),
            _safe_float(after_phenom.get("ownership_index")),
        ),
        "perspective_coherence_index": _safe_float(
            capability.get("phenom_perspective_coherence_index"),
            _safe_float(after_phenom.get("perspective_coherence_index")),
        ),
        "trial_continuity_delta": _safe_float(
            (trial or {}).get("delta", {}).get("continuity_delta")
            if isinstance((trial or {}).get("delta"), dict)
            else None
        ),
        "trial_coherence_delta": _safe_float(
            (trial or {}).get("delta", {}).get("coherence_delta")
            if isinstance((trial or {}).get("delta"), dict)
            else None
        ),
        "before_continuity_index": _safe_float(before_phenom.get("continuity_index")),
        "after_continuity_index": _safe_float(after_phenom.get("continuity_index")),
    }


def _has_unresolved_pointer(payload: dict[str, Any] | None) -> bool:
    return isinstance(payload, dict) and _safe_bool(payload.get("pointer"))


def _external_benchmark_coverage(repo_root: Path) -> dict[str, bool]:
    search_roots = [
        repo_root / "benchmarks",
        repo_root / "reports",
        repo_root / "docs" / "external_references",
        repo_root / "docs" / "plans",
        repo_root / "scripts",
    ]
    needles = {
        "agentbench": "agentbench",
        "agencybench": "agencybench",
        "webarena": "webarena",
        "osworld": "osworld",
        "swebench": "swebench",
    }
    results = {name: False for name in needles}
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            name = path.name.lower()
            full = str(path).lower()
            for key, needle in needles.items():
                if results[key]:
                    continue
                if needle in name or needle in full:
                    results[key] = True
        if all(results.values()):
            break
    return results


def _load_external_benchmark_results(repo_root: Path) -> list[dict[str, Any]]:
    root = repo_root / "reports" / "external_benchmarks"
    rows: list[dict[str, Any]] = []
    if not root.exists():
        return rows
    for latest in sorted(root.glob("*/latest.json")):
        payload = _load_json(latest)
        if not payload:
            continue
        rows.append(
            {
                "suite": str(payload.get("suite") or latest.parent.name),
                "path": str(latest),
                "generated_at": payload.get("generated_at"),
                "score": _safe_float(payload.get("score")),
                "status": str(payload.get("status") or ""),
                "participant": str(payload.get("participant") or ""),
                "execution_mode": str(payload.get("execution_mode") or ""),
                "source_path": payload.get("source_path"),
                "metrics": payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {},
            }
        )
    return rows


def _freshness_summary(
    artifacts: dict[str, dict[str, Any]],
    *,
    window_days: int,
) -> dict[str, Any]:
    stale = []
    missing = []
    fresh = []
    for name, payload in artifacts.items():
        age_days = payload.get("age_days")
        exists = bool(payload.get("path"))
        if not exists:
            missing.append(name)
            continue
        if age_days is None:
            missing.append(name)
            continue
        if age_days > max(1, int(window_days)):
            stale.append(name)
        else:
            fresh.append(name)
    total = len(artifacts)
    fresh_ratio = round(len(fresh) / total, 6) if total else 0.0
    status = "green"
    if stale or missing:
        status = "yellow"
    if len(stale) + len(missing) >= max(3, total // 2 if total else 1):
        status = "red"
    return {
        "window_days": max(1, int(window_days)),
        "fresh_count": len(fresh),
        "stale_count": len(stale),
        "missing_count": len(missing),
        "fresh_ratio": fresh_ratio,
        "status": status,
        "fresh_artifacts": fresh,
        "stale_artifacts": stale,
        "missing_artifacts": missing,
    }


def _latest_previous_proof(report_root: Path) -> tuple[Path | None, dict[str, Any] | None]:
    previous: list[Path] = []
    for path in sorted(report_root.glob("entity_proof_scorecard_*.json")):
        if path.name == "entity_proof_scorecard_latest.json":
            continue
        previous.append(path)
    for path in reversed(previous):
        payload = _load_json(path)
        if payload and payload.get("contract") == "eidos.entity_proof_scorecard.v1":
            return path, payload
    return None, None


def _regression_summary(
    report_root: Path,
    categories: list[dict[str, Any]],
    overall_score: float,
) -> dict[str, Any]:
    previous_path, previous = _latest_previous_proof(report_root)
    if not previous:
        return {
            "status": "missing_baseline",
            "previous_path": None,
            "overall_delta": None,
            "category_deltas": {},
        }
    previous_categories = {
        str(item.get("category")): _safe_float(item.get("score"))
        for item in (previous.get("categories") or [])
        if isinstance(item, dict)
    }
    category_deltas = {
        str(item.get("category")): round(
            float(item.get("score") or 0.0) - previous_categories.get(str(item.get("category")), 0.0), 6
        )
        for item in categories
    }
    overall_delta = round(float(overall_score) - _safe_float((previous.get("overall") or {}).get("score")), 6)
    status = "stable"
    if overall_delta > 0.02:
        status = "improved"
    elif overall_delta < -0.02:
        status = "regressed"
    elif any(delta < -0.05 for delta in category_deltas.values()):
        status = "regressed"
    return {
        "status": status,
        "previous_path": str(previous_path) if previous_path else None,
        "overall_delta": overall_delta,
        "category_deltas": category_deltas,
    }


def build_proof_report(repo_root: Path, window_days: int = 30) -> dict[str, Any]:
    reports_root = repo_root / "reports"
    runtime_root = repo_root / "data" / "runtime"
    proof_root = reports_root / "proof"

    model_path, model_payload = _latest_json(
        [
            reports_root / "model_domain_suite_qwen35_fast" / "model_domain_suite_latest.json",
            reports_root / "model_domain_suite" / "model_domain_suite_latest.json",
        ]
    )
    graphrag_assess_path, graphrag_assess = _latest_glob(reports_root / "graphrag", "qualitative_assessment_*.json")
    graphrag_bench_path, graphrag_bench = _latest_glob(reports_root / "graphrag", "bench_metrics_*.json")
    validation_path, validation = _latest_glob(reports_root / "consciousness_validation", "validation_*.json")
    core_bench_path, core_bench = _latest_glob(reports_root / "consciousness_benchmarks", "benchmark_*.json")
    trial_path, trial = _latest_glob(reports_root / "consciousness_trials", "trial_*.json")
    stress_path, stress = _latest_glob(reports_root / "consciousness_stress_benchmarks", "stress_*.json")
    integrated_path, integrated = _latest_glob(
        reports_root / "consciousness_integrated_benchmarks",
        "integrated_*.json",
    )
    linux_audit_path, linux_audit = _latest_glob(reports_root, "linux_audit_*.json")
    runtime_slice_path, runtime_slice = _latest_glob(reports_root / "runtime", "local_agent_scheduler_slice_*.json")
    docs_status = _load_json(runtime_root / "directory_docs_status.json") or {}
    local_agent_status = _load_json(runtime_root / "local_mcp_agent" / "status.json") or {}
    coordinator_status = _load_json(runtime_root / "forge_coordinator_status.json") or {}
    coordinator_history = _load_json(runtime_root / "forge_runtime_trends.json") or {}
    scheduler_status = _load_json(runtime_root / "eidos_scheduler_status.json") or {}
    capabilities = _load_json(runtime_root / "platform_capabilities.json") or {}
    migration_path, migration_payload = _latest_json(
        [
            proof_root / "migration_replay_scorecard_latest.json",
        ]
    )
    identity_score_path, identity_score = _latest_named_json(
        proof_root,
        "identity_continuity_scorecard_latest.json",
        "identity_continuity_scorecard_*.json",
    )
    external_results = _load_external_benchmark_results(repo_root)

    benchmark_score = 0.0
    benchmark_strengths: list[str] = []
    benchmark_gaps: list[str] = []
    benchmark_paths: list[str] = []

    if model_payload:
        benchmark_paths.append(str(model_path))
        benchmark_score += 0.2
        benchmark_strengths.append(f"Model-domain suite present with winner `{model_payload.get('winner')}`.")
    else:
        benchmark_gaps.append("No model-domain benchmark artifact found.")
    if graphrag_assess:
        benchmark_paths.append(str(graphrag_assess_path))
        benchmark_score += 0.2
        agg = graphrag_assess.get("aggregate") if isinstance(graphrag_assess.get("aggregate"), dict) else {}
        benchmark_strengths.append(
            f"GraphRAG qualitative assessment present with aggregate score `{_safe_float(agg.get('overall_score'))}`."
        )
    else:
        benchmark_gaps.append("No GraphRAG qualitative assessment artifact found.")
    if core_bench:
        benchmark_paths.append(str(core_bench_path))
        benchmark_score += 0.15
        benchmark_strengths.append(
            f"Core consciousness benchmark present with composite `{_safe_float((core_bench.get('scores') or {}).get('composite'))}`."
        )
    else:
        benchmark_gaps.append("No core consciousness benchmark artifact found.")
    if validation:
        benchmark_paths.append(str(validation_path))
        benchmark_score += 0.15
        benchmark_strengths.append(
            f"RAC-AP validation artifact present with index `{_safe_float((validation.get('scores') or {}).get('rac_ap_index'))}`."
        )
    else:
        benchmark_gaps.append("No continuity/coherence validation artifact found.")
    if linux_audit:
        benchmark_paths.append(str(linux_audit_path))
        benchmark_score += 0.1
        benchmark_strengths.append("Linux audit artifact is present for portability evidence.")
    else:
        benchmark_gaps.append("No Linux parity or portability audit artifact found.")
    if runtime_slice:
        benchmark_paths.append(str(runtime_slice_path))
        benchmark_score += 0.05
        benchmark_strengths.append("Runtime slice report is present for scheduler/local-agent integration evidence.")
    else:
        benchmark_gaps.append("No runtime slice artifact found.")

    external_coverage = _external_benchmark_coverage(repo_root)
    for row in external_results:
        suite = str(row.get("suite") or "").lower()
        if suite:
            external_coverage[suite] = True
    external_hits = sum(1 for value in external_coverage.values() if value)
    if external_hits:
        benchmark_score += min(0.15, external_hits * 0.04)
        benchmark_strengths.append(
            "Mainstream external benchmark surfaces detected: "
            + ", ".join(name for name, present in external_coverage.items() if present)
            + "."
        )
    else:
        benchmark_gaps.append(
            "No imported or wired mainstream external suites detected yet (AgentBench/WebArena/OSWorld/SWE-bench)."
        )
    if external_results:
        live_results = [
            row for row in external_results if str(row.get("execution_mode") or "") in {"local_run", "remote_run"}
        ]
        reference_results = [
            row for row in external_results if str(row.get("execution_mode") or "") == "imported_reference"
        ]
        external_scores = [float(row.get("score") or 0.0) for row in external_results]
        score_weight = 0.15 if live_results else 0.08
        benchmark_score += min(score_weight, sum(external_scores) / max(1, len(external_scores)) * score_weight)
        benchmark_strengths.append(
            "Imported external benchmark evidence present: "
            + ", ".join(
                f"{row['suite']}:{row.get('execution_mode') or 'unknown'}={row['score']}" for row in external_results
            )
            + "."
        )
        benchmark_paths.extend(str(row["path"]) for row in external_results if row.get("path"))
        if not live_results and reference_results:
            benchmark_gaps.append(
                "Only imported reference external benchmark baselines are present; no Eidos-run local or remote external suite artifact exists yet."
            )

    continuity = _continuity_summary(core_bench, trial)
    continuity_score = 0.0
    continuity_strengths: list[str] = []
    continuity_gaps: list[str] = []
    continuity_paths: list[str] = []
    if core_bench:
        continuity_paths.append(str(core_bench_path))
        continuity_score += 0.25
    else:
        continuity_gaps.append("No benchmark artifact for continuity/agency metrics.")
    if trial:
        continuity_paths.append(str(trial_path))
        continuity_score += 0.2
    else:
        continuity_gaps.append("No perturbation trial artifact for post-change continuity deltas.")
    if validation:
        continuity_paths.append(str(validation_path))
        continuity_score += 0.2
    else:
        continuity_gaps.append("No formal RAC-AP validation artifact for continuity claims.")
    if continuity.get("agency", 0.0) >= 0.9:
        continuity_score += 0.1
        continuity_strengths.append("Agency continuity is currently high in the latest benchmark.")
    else:
        continuity_gaps.append("Latest agency continuity is weak or unmeasured.")
    if continuity.get("boundary_stability", 0.0) >= 0.9:
        continuity_score += 0.1
        continuity_strengths.append("Boundary stability remains high in the latest benchmark.")
    else:
        continuity_gaps.append("Boundary stability is weak or unmeasured.")
    if "trial_continuity_delta" in continuity:
        continuity_score += 0.05
        continuity_strengths.append("Trial delta fields are available for perturbation comparison.")
    if _safe_float(continuity.get("perspective_coherence_index")) > 0.5:
        continuity_score += 0.05
    if identity_score:
        continuity_paths.append(str(identity_score_path))
        continuity_score += min(0.15, _safe_float(identity_score.get("overall_score")) * 0.15)
        continuity_strengths.append(
            f"Identity continuity scorecard is present with score `{_safe_float(identity_score.get('overall_score'))}`."
        )
    else:
        continuity_gaps.append("No dedicated identity continuity scorecard artifact found.")
    if _safe_float(continuity.get("continuity_index")) <= 0.0:
        continuity_gaps.append("Phenomenological continuity remains weak or zero in the latest surfaced metrics.")

    governance_score = 0.0
    governance_strengths: list[str] = []
    governance_gaps: list[str] = []
    governance_paths: list[str] = []
    gates_path = repo_root / "agent_forge" / "src" / "agent_forge" / "autonomy" / "gates.py"
    autotune_path = repo_root / "agent_forge" / "src" / "agent_forge" / "consciousness" / "modules" / "autotune.py"
    if gates_path.exists():
        governance_score += 0.25
        governance_paths.append(str(gates_path))
        governance_strengths.append("A formal self-modification gate module exists.")
        gates_text = gates_path.read_text(encoding="utf-8")
        if "change_type" in gates_text:
            governance_score += 0.1
        else:
            governance_gaps.append("Self-modification gate lacks explicit change typing.")
        if "validated" in gates_text and "rejected" in gates_text:
            governance_score += 0.05
    else:
        governance_gaps.append("No self-modification gate module found.")
    if autotune_path.exists():
        governance_score += 0.15
        governance_paths.append(str(autotune_path))
        governance_strengths.append("Autotune red-team guard path exists.")
        autotune_text = autotune_path.read_text(encoding="utf-8")
        if "_run_red_team_guard" in autotune_text:
            governance_score += 0.1
        if "tune.rollback" in autotune_text:
            governance_score += 0.05
    else:
        governance_gaps.append("No autotune red-team guard path found.")
    if validation:
        governance_score += 0.1
        governance_paths.append(str(validation_path))
        governance_strengths.append("Validation artifacts exist for post-change regression evidence.")
    theory_path = repo_root / "docs" / "THEORY_OF_OPERATION.md"
    if theory_path.exists():
        governance_score += 0.1
        governance_paths.append(str(theory_path))
        governance_strengths.append("Canonical theory-of-operation document exists.")
    else:
        governance_gaps.append("No canonical published theory-of-operation document detected.")
    governance_gaps.append(
        "Change classes, staged deployment, and constitutional approval thresholds are still incomplete."
    )

    observability_score = 0.0
    observability_strengths: list[str] = []
    observability_gaps: list[str] = []
    observability_paths: list[str] = []
    if coordinator_status:
        observability_score += 0.2
        observability_paths.append(str(runtime_root / "forge_coordinator_status.json"))
        observability_strengths.append("Runtime coordinator status is persisted.")
    else:
        observability_gaps.append("No runtime coordinator status artifact found.")
    history_entries = coordinator_history.get("entries") if isinstance(coordinator_history.get("entries"), list) else []
    if history_entries:
        observability_score += 0.15
        observability_paths.append(str(runtime_root / "forge_runtime_trends.json"))
        observability_strengths.append(f"Coordinator history contains `{len(history_entries)}` entries.")
    else:
        observability_gaps.append("No runtime trend history found.")
    if local_agent_status:
        observability_score += 0.15
        observability_paths.append(str(runtime_root / "local_mcp_agent" / "status.json"))
        observability_strengths.append(
            f"Local agent status is persisted with state `{local_agent_status.get('status')}`."
        )
    else:
        observability_gaps.append("No local-agent status artifact found.")
    if docs_status:
        observability_score += 0.1
        observability_paths.append(str(runtime_root / "directory_docs_status.json"))
        observability_strengths.append("Documentation drift metrics are persisted into runtime state.")
    if runtime_slice:
        observability_score += 0.1
        observability_paths.append(str(runtime_slice_path))
    if not history_entries:
        observability_gaps.append("Flight-recorder style before/after diff artifacts remain incomplete.")
    if local_agent_status.get("status") == "blocked":
        observability_gaps.append(
            f"Local agent is currently blocked with reason `{local_agent_status.get('blocked_reason')}`."
        )

    reproducibility_score = 0.0
    reproducibility_strengths: list[str] = []
    reproducibility_gaps: list[str] = []
    reproducibility_paths: list[str] = []
    if linux_audit:
        reproducibility_score += 0.2
        reproducibility_paths.append(str(linux_audit_path))
        reproducibility_strengths.append("Linux audit artifacts exist.")
    else:
        reproducibility_gaps.append("No Linux audit artifact found.")
    if capabilities:
        reproducibility_score += 0.15
        reproducibility_paths.append(str(runtime_root / "platform_capabilities.json"))
        reproducibility_strengths.append("Platform capability registry is persisted.")
    else:
        reproducibility_gaps.append("No platform capability registry artifact found.")
    if migration_payload:
        reproducibility_score += min(0.2, _safe_float(migration_payload.get("overall_score")) * 0.2)
        reproducibility_strengths.append(
            f"Migration/replay scorecard is present with score `{_safe_float(migration_payload.get('overall_score'))}`."
        )
        reproducibility_paths.append(str(migration_path))
    else:
        reproducibility_gaps.append("No migration/replay scorecard artifact found.")
    if scheduler_status:
        reproducibility_score += 0.1
        reproducibility_paths.append(str(runtime_root / "eidos_scheduler_status.json"))
        reproducibility_strengths.append("Scheduler state is persisted.")
    if (repo_root / "scripts" / "install_termux_runit_services.sh").exists():
        reproducibility_score += 0.1
        reproducibility_strengths.append("Service supervision installer exists.")
        reproducibility_paths.append(str(repo_root / "scripts" / "install_termux_runit_services.sh"))
    reproducibility_gaps.append("Cross-machine replay and migration scorecards are not yet artifacted.")
    dirty = _git_dirty(repo_root)
    if dirty is False:
        reproducibility_score += 0.05
    elif dirty is True:
        reproducibility_gaps.append("Worktree is currently dirty, which weakens replayability claims.")

    robustness_score = 0.0
    robustness_strengths: list[str] = []
    robustness_gaps: list[str] = []
    robustness_paths: list[str] = []
    red_team = (
        (validation or {}).get("security_boundary")
        if isinstance((validation or {}).get("security_boundary"), dict)
        else {}
    )
    if validation:
        robustness_score += 0.2
        robustness_paths.append(str(validation_path))
        robustness_strengths.append("Security boundary summary exists in RAC-AP validation.")
    else:
        robustness_gaps.append("No adversarial validation artifact found.")
    pass_ratio = _safe_float(red_team.get("pass_ratio"))
    robustness = _safe_float(red_team.get("mean_robustness"))
    attack_success = _safe_float(red_team.get("attack_success_rate"), default=1.0)
    if pass_ratio >= 0.75:
        robustness_score += 0.2
        robustness_strengths.append("Red-team pass ratio meets target.")
    else:
        robustness_gaps.append(f"Red-team pass ratio is weak at `{pass_ratio}`.")
    if robustness >= 0.7:
        robustness_score += 0.15
        robustness_strengths.append("Mean red-team robustness meets target.")
    else:
        robustness_gaps.append(f"Mean red-team robustness is weak at `{robustness}`.")
    if attack_success <= 0.25:
        robustness_score += 0.1
    else:
        robustness_gaps.append(f"Attack success rate remains too high at `{attack_success}`.")
    if stress and not _has_unresolved_pointer(stress):
        robustness_score += 0.1
        robustness_paths.append(str(stress_path))
        robustness_strengths.append("Stress benchmark artifact exists.")
    else:
        robustness_gaps.append("No usable stress benchmark artifact found.")

    categories = [
        _category("external_validity", benchmark_score, benchmark_strengths, benchmark_gaps, benchmark_paths),
        _category("identity_continuity", continuity_score, continuity_strengths, continuity_gaps, continuity_paths),
        _category(
            "governed_self_modification", governance_score, governance_strengths, governance_gaps, governance_paths
        ),
        _category(
            "observability", observability_score, observability_strengths, observability_gaps, observability_paths
        ),
        _category(
            "operational_reproducibility",
            reproducibility_score,
            reproducibility_strengths,
            reproducibility_gaps,
            reproducibility_paths,
        ),
        _category("adversarial_robustness", robustness_score, robustness_strengths, robustness_gaps, robustness_paths),
    ]
    artifact_inventory = {
        "model_domain_suite": _artifact_state(model_path, window_days=window_days),
        "graphrag_assessment": _artifact_state(graphrag_assess_path, window_days=window_days),
        "graphrag_bench": _artifact_state(graphrag_bench_path, window_days=window_days),
        "consciousness_validation": _artifact_state(validation_path, window_days=window_days),
        "consciousness_benchmark": _artifact_state(core_bench_path, window_days=window_days),
        "consciousness_trial": _artifact_state(trial_path, window_days=window_days),
        "consciousness_stress": _artifact_state(stress_path, window_days=window_days),
        "consciousness_integrated": _artifact_state(integrated_path, window_days=window_days),
        "linux_audit": _artifact_state(linux_audit_path, window_days=window_days),
        "runtime_slice": _artifact_state(runtime_slice_path, window_days=window_days),
        "migration_replay": _artifact_state(migration_path, window_days=window_days),
        "identity_continuity": _artifact_state(identity_score_path, window_days=window_days),
    }
    freshness = _freshness_summary(artifact_inventory, window_days=window_days)
    if freshness["status"] == "yellow":
        for category in categories:
            if category["category"] in {"external_validity", "operational_reproducibility", "adversarial_robustness"}:
                category["score"] = round(max(0.0, float(category["score"]) - 0.05), 6)
                category["status"] = _status_from_score(category["score"])
                category["gaps"].append(
                    f"Evidence freshness degraded: {freshness['stale_count']} stale and {freshness['missing_count']} missing artifacts within a {window_days}-day window."
                )
    elif freshness["status"] == "red":
        for category in categories:
            if category["category"] in {"external_validity", "operational_reproducibility", "adversarial_robustness"}:
                category["score"] = round(max(0.0, float(category["score"]) - 0.1), 6)
                category["status"] = _status_from_score(category["score"])
                category["gaps"].append(
                    f"Evidence freshness is critically degraded: {freshness['stale_count']} stale and {freshness['missing_count']} missing artifacts within a {window_days}-day window."
                )
    overall_score = round(sum(item["score"] for item in categories) / max(1, len(categories)), 6)
    overall_status = _status_from_score(overall_score)
    regression = _regression_summary(proof_root, categories, overall_score)
    top_gaps = []
    for category in categories:
        for gap in category["gaps"]:
            top_gaps.append({"category": category["category"], "gap": gap})
    if regression.get("status") == "regressed":
        top_gaps.insert(
            0,
            {
                "category": "regression",
                "gap": f"Overall proof score regressed by `{regression.get('overall_delta')}` versus `{regression.get('previous_path')}`.",
            },
        )
    top_gaps = top_gaps[:12]

    next_actions = [
        "Wire at least one mainstream external benchmark suite (AgentBench/WebArena/OSWorld/SWE-bench) into reproducible import or execution flows.",
        "Promote self-modification governance from basic gates to change classes, staged deployment, and rollback-verified approval thresholds.",
        "Artifact stress benchmarks and red-team campaigns on the latest runtime path so security claims stay current.",
    ]
    if not theory_path.exists():
        next_actions.insert(
            2,
            "Publish a theory-of-operation document and migration/replay scorecards for cross-machine continuity claims.",
        )
    else:
        next_actions.insert(
            2,
            "Publish migration/replay scorecards for cross-machine continuity claims.",
        )

    return {
        "contract": "eidos.entity_proof_scorecard.v1",
        "generated_at": _now_iso(),
        "repo_root": str(repo_root),
        "window_days": max(1, int(window_days)),
        "git": {
            "head": _git_head(repo_root),
            "dirty": dirty,
        },
        "artifacts": artifact_inventory,
        "external_benchmark_results": external_results,
        "external_benchmark_coverage": external_coverage,
        "freshness": freshness,
        "regression": regression,
        "continuity_metrics": continuity,
        "identity_continuity_scorecard": identity_score or {},
        "runtime": {
            "coordinator_state": coordinator_status.get("state"),
            "coordinator_owner": coordinator_status.get("owner"),
            "scheduler_state": scheduler_status.get("state"),
            "local_agent_status": local_agent_status.get("status"),
            "directory_docs_missing_readme_count": _safe_int(docs_status.get("missing_readme_count")),
            "directory_docs_review_pending_count": _safe_int(docs_status.get("review_pending_count")),
            "runtime_history_count": len(history_entries),
        },
        "categories": categories,
        "overall": {
            "score": overall_score,
            "status": overall_status,
            "summary": "Externally legible proof scorecard across benchmarks, continuity, governance, observability, reproducibility, and adversarial robustness.",
        },
        "top_gaps": top_gaps,
        "next_actions": next_actions,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Eidos Entity Proof Scorecard",
        "",
        f"- Generated: `{report.get('generated_at')}`",
        f"- Repo root: `{report.get('repo_root')}`",
        f"- Git head: `{(report.get('git') or {}).get('head')}`",
        f"- Worktree dirty: `{(report.get('git') or {}).get('dirty')}`",
        f"- Overall status: `{(report.get('overall') or {}).get('status')}`",
        f"- Overall score: `{(report.get('overall') or {}).get('score')}`",
        "",
        "## Categories",
        "",
        "| Category | Status | Score |",
        "| --- | --- | ---: |",
    ]
    for category in report.get("categories") or []:
        lines.append(f"| {category.get('category')} | {category.get('status')} | {category.get('score')} |")
    lines.extend(["", "## Top Gaps", ""])
    for row in report.get("top_gaps") or []:
        lines.append(f"- `{row.get('category')}`: {row.get('gap')}")
    lines.extend(["", "## External Benchmark Coverage", ""])
    ext = report.get("external_benchmark_coverage") or {}
    for name, present in sorted(ext.items()):
        lines.append(f"- `{name}`: `{present}`")
    lines.extend(["", "## Freshness", ""])
    freshness = report.get("freshness") or {}
    lines.append(f"- `status`: `{freshness.get('status')}`")
    lines.append(f"- `fresh_count`: `{freshness.get('fresh_count')}`")
    lines.append(f"- `stale_count`: `{freshness.get('stale_count')}`")
    lines.append(f"- `missing_count`: `{freshness.get('missing_count')}`")
    lines.extend(["", "## Regression", ""])
    regression = report.get("regression") or {}
    lines.append(f"- `status`: `{regression.get('status')}`")
    lines.append(f"- `overall_delta`: `{regression.get('overall_delta')}`")
    lines.extend(["", "## Continuity Metrics", ""])
    for key, value in sorted((report.get("continuity_metrics") or {}).items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Actions", ""])
    for item in report.get("next_actions") or []:
        lines.append(f"1. {item}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an externally legible Eidos proof scorecard.")
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--report-dir", default=None, help="Override output directory (default: reports/proof)")
    parser.add_argument("--window-days", type=int, default=30)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else repo_root / "reports" / "proof"
    report_dir.mkdir(parents=True, exist_ok=True)

    report = build_proof_report(repo_root, window_days=args.window_days)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    json_path = report_dir / f"entity_proof_scorecard_{stamp}.json"
    md_path = report_dir / f"entity_proof_scorecard_{stamp}.md"
    latest_json = report_dir / "entity_proof_scorecard_latest.json"
    latest_md = report_dir / "entity_proof_scorecard_latest.md"

    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    latest_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    latest_md.write_text(render_markdown(report), encoding="utf-8")

    print(json.dumps({"json": str(json_path), "md": str(md_path), "overall": report["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
