#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str
    duration_ms: float


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


def _json_loads(text: str) -> dict[str, Any] | None:
    try:
        payload = json.loads(text)
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    return dict(payload)


def _extract_report_path(stdout_text: str) -> Path | None:
    match = re.search(r"Report:\s*(.+)", stdout_text)
    if not match:
        return None
    path = Path(match.group(1).strip()).expanduser().resolve()
    return path


def _run_cmd(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    timeout_s: float,
    cwd: Path,
) -> tuple[int, str, str, float]:
    started = time.perf_counter()
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            env=dict(env),
            cwd=str(cwd),
            timeout=max(1.0, float(timeout_s)),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = (time.perf_counter() - started) * 1000.0
        stdout = str(getattr(exc, "stdout", "") or "")
        stderr = str(getattr(exc, "stderr", "") or "")
        return 124, stdout, stderr or "timeout", duration_ms
    duration_ms = (time.perf_counter() - started) * 1000.0
    return int(proc.returncode), str(proc.stdout or ""), str(proc.stderr or ""), duration_ms


def evaluate_forge_status(payload: Mapping[str, Any]) -> tuple[bool, str]:
    optional_forges = {
        name.strip()
        for name in os.environ.get("EIDOS_AUDIT_OPTIONAL_FORGES", "mkey").split(",")
        if name.strip()
    }
    forges = payload.get("forges")
    if not isinstance(forges, Mapping):
        return False, "missing forges payload"
    bad: list[str] = []
    for name, info in forges.items():
        if not isinstance(info, Mapping):
            bad.append(str(name))
            continue
        available = bool(info.get("available"))
        status = str(info.get("status") or "")
        if ((not available) or status == "error") and str(name) not in optional_forges:
            bad.append(str(name))
    if bad:
        return False, f"unavailable/error forges: {', '.join(sorted(bad))}"
    if optional_forges:
        return True, f"all required forges available (optional skipped: {', '.join(sorted(optional_forges))})"
    return True, "all forges available"


def evaluate_mcp_audit(payload: Mapping[str, Any], *, strict: bool) -> tuple[bool, str]:
    counts = payload.get("counts")
    if not isinstance(counts, Mapping):
        return False, "missing counts payload"
    tool_hard_fail = _safe_int(counts.get("tool_hard_fail"))
    resource_hard_fail = _safe_int(counts.get("resource_hard_fail"))
    tool_soft_fail = _safe_int(counts.get("tool_soft_fail"))
    if tool_hard_fail > 0 or resource_hard_fail > 0:
        return (
            False,
            f"hard fails tool={tool_hard_fail} resource={resource_hard_fail}",
        )
    if strict and tool_soft_fail > 0:
        return False, f"strict mode soft fails tool={tool_soft_fail}"
    return True, f"ok hard_fail=0 soft_fail={tool_soft_fail}"


def evaluate_consciousness_status(payload: Mapping[str, Any]) -> tuple[bool, str]:
    required = ("workspace", "coherence", "rci", "watchdog", "payload_safety")
    missing = [key for key in required if key not in payload]
    if missing:
        return False, f"missing keys: {', '.join(missing)}"
    watchdog = payload.get("watchdog")
    if not isinstance(watchdog, Mapping):
        return False, "invalid watchdog payload"
    if "enabled" not in watchdog:
        return False, "watchdog.enabled missing"
    return True, "status payload valid"


def evaluate_benchmark(payload: Mapping[str, Any]) -> tuple[bool, str]:
    scores = payload.get("scores")
    gates = payload.get("gates")
    if not isinstance(scores, Mapping) or not isinstance(gates, Mapping):
        return False, "missing scores/gates payload"
    composite = _safe_float(scores.get("composite"), default=-1.0)
    if composite < 0.0:
        return False, "invalid composite score"
    required_gates = ("world_model_online", "meta_online", "report_online")
    off = [key for key in required_gates if not bool(gates.get(key))]
    if off:
        return False, f"required gates failed: {', '.join(off)}"
    return True, f"composite={composite}"


def evaluate_stress_benchmark(
    payload: Mapping[str, Any],
    *,
    strict_latency: bool = False,
) -> tuple[bool, str]:
    pressure = payload.get("pressure")
    gates = payload.get("gates")
    if not isinstance(pressure, Mapping) or not isinstance(gates, Mapping):
        return False, "missing pressure/gates payload"
    truncations = _safe_int(pressure.get("payload_truncations_observed"))
    if truncations <= 0:
        return False, "payload truncation not observed"
    required = ("event_pressure_hits_target", "module_error_free")
    off = [key for key in required if not bool(gates.get(key))]
    if off:
        return False, f"stress gates failed: {', '.join(off)}"
    latency_ok = bool(gates.get("latency_p95_under_200ms"))
    if strict_latency and (not latency_ok):
        return False, "stress latency gate failed: latency_p95_under_200ms"
    if not latency_ok:
        return True, f"truncations={truncations} latency_gate=warn"
    return True, f"truncations={truncations}"


def evaluate_full_benchmark(payload: Mapping[str, Any]) -> tuple[bool, str]:
    scores = payload.get("scores")
    gates = payload.get("gates")
    if not isinstance(scores, Mapping) or not isinstance(gates, Mapping):
        return False, "missing scores/gates payload"
    integrated = _safe_float(scores.get("integrated"), default=-1.0)
    if integrated < 0.0:
        return False, "invalid integrated score"
    required = ("core_score_min", "trial_score_min", "non_regression")
    off = [key for key in required if not bool(gates.get(key))]
    if off:
        return False, f"full benchmark gates failed: {', '.join(off)}"
    return True, f"integrated={integrated}"


def _python_bin(root: Path, override: str | None) -> str:
    if override:
        return override
    venv_py = root / "eidosian_venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return str(Path(sys.executable))


async def _run_mcp_quick_probe(
    *,
    root: Path,
    python_bin: str,
    env: Mapping[str, str],
    state_dir: Path,
    timeout_s: float,
) -> dict[str, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except Exception as exc:
        return {
            "counts": {
                "tools_total": 0,
                "resources_total": 0,
                "tool_ok": 0,
                "tool_soft_fail": 0,
                "tool_hard_fail": 1,
                "tool_skipped": 0,
                "resource_ok": 0,
                "resource_hard_fail": 0,
            },
            "error": f"mcp-client-import-failed: {exc}",
        }

    params = StdioServerParameters(
        command=python_bin,
        args=["-u", "-c", "import eidos_mcp.eidos_mcp_server as s; s.main()"],
        env={
            **dict(env),
            "EIDOS_MCP_TRANSPORT": "stdio",
            "EIDOS_MCP_STATELESS_HTTP": "1",
            "EIDOS_FORGE_DIR": str(root),
        },
    )
    tool_ok = 0
    tool_soft_fail = 0
    tool_hard_fail = 0
    resource_hard_fail = 0
    resources_total = 0
    tools_total = 2
    outcomes: list[dict[str, Any]] = []

    init_timeout = max(30.0, float(timeout_s))
    call_timeout = max(12.0, float(timeout_s))

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await asyncio.wait_for(session.initialize(), timeout=init_timeout)
            resources = await asyncio.wait_for(session.list_resources(), timeout=call_timeout)
            resources_total = len(resources.resources)
            resource_ok = resources_total
            calls = [
                ("diagnostics_ping", {}),
                ("consciousness_kernel_status", {"state_dir": str(state_dir)}),
            ]
            for name, arguments in calls:
                try:
                    result = await asyncio.wait_for(
                        session.call_tool(name, arguments=arguments),
                        timeout=call_timeout,
                    )
                    text = ""
                    if result.structuredContent and "result" in result.structuredContent:
                        text = str(result.structuredContent["result"])
                    elif result.content:
                        for content in result.content:
                            if getattr(content, "type", None) == "text":
                                text = str(content.text or "")
                                break
                    if getattr(result, "isError", False):
                        tool_soft_fail += 1
                        outcomes.append({"tool": name, "status": "soft_fail", "sample": text[:180]})
                    else:
                        tool_ok += 1
                        outcomes.append({"tool": name, "status": "ok", "sample": text[:180]})
                except Exception as exc:
                    tool_hard_fail += 1
                    outcomes.append({"tool": name, "status": "hard_fail", "error": str(exc)})

    return {
        "counts": {
            "tools_total": tools_total,
            "resources_total": resources_total,
            "tool_ok": tool_ok,
            "tool_soft_fail": tool_soft_fail,
            "tool_hard_fail": tool_hard_fail,
            "tool_skipped": 0,
            "resource_ok": resource_ok if resources_total >= 0 else 0,
            "resource_hard_fail": resource_hard_fail,
        },
        "tool_outcomes": outcomes,
    }


def run_matrix(
    *,
    root: Path,
    python_bin: str,
    state_dir: Path,
    report_dir: Path,
    timeout_s: float,
    mcp_timeout_s: float,
    quick: bool,
    strict: bool,
) -> dict[str, Any]:
    run_id = f"linux_audit_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    report_dir.mkdir(parents=True, exist_ok=True)

    py_path = ":".join(
        [
            str(root / "lib"),
            str(root / "agent_forge" / "src"),
            str(root / "eidos_mcp" / "src"),
            str(root / "crawl_forge" / "src"),
        ]
    )
    env = {
        **os.environ,
        "PYTHONPATH": py_path,
        "EIDOS_FORGE_DIR": str(root),
        "PYTHONUNBUFFERED": "1",
    }

    checks: list[CheckResult] = []
    failures: list[str] = []

    def run_json_check(
        *,
        name: str,
        cmd: Sequence[str],
        evaluator,
        timeout_override: float | None = None,
    ) -> None:
        rc, out, err, dur = _run_cmd(
            cmd,
            env=env,
            timeout_s=timeout_override if timeout_override is not None else timeout_s,
            cwd=root,
        )
        if rc != 0:
            details = f"rc={rc} stderr={err.strip()[:260]}"
            checks.append(CheckResult(name=name, ok=False, details=details, duration_ms=round(dur, 6)))
            failures.append(name)
            return
        payload = _json_loads(out)
        if payload is None:
            details = "invalid JSON output"
            checks.append(CheckResult(name=name, ok=False, details=details, duration_ms=round(dur, 6)))
            failures.append(name)
            return
        ok, details = evaluator(payload)
        checks.append(CheckResult(name=name, ok=ok, details=details, duration_ms=round(dur, 6)))
        if not ok:
            failures.append(name)

    run_json_check(
        name="forge_status",
        cmd=[python_bin, str(root / "bin" / "eidosian"), "--json", "status"],
        evaluator=evaluate_forge_status,
    )

    if quick:
        mcp_started = time.perf_counter()
        try:
            payload = asyncio.run(
                _run_mcp_quick_probe(
                    root=root,
                    python_bin=python_bin,
                    env=env,
                    state_dir=state_dir,
                    timeout_s=max(2.0, float(mcp_timeout_s)),
                )
            )
            ok, details = evaluate_mcp_audit(payload, strict=strict)
            checks.append(
                CheckResult(
                    name="mcp_audit_matrix",
                    ok=ok,
                    details=f"{details} mode=quick_probe",
                    duration_ms=round((time.perf_counter() - mcp_started) * 1000.0, 6),
                )
            )
            if not ok:
                failures.append("mcp_audit_matrix")
        except Exception as exc:
            checks.append(
                CheckResult(
                    name="mcp_audit_matrix",
                    ok=False,
                    details=f"quick_probe_error={exc}",
                    duration_ms=round((time.perf_counter() - mcp_started) * 1000.0, 6),
                )
            )
            failures.append("mcp_audit_matrix")
    else:
        mcp_cmd = [
            python_bin,
            str(root / "scripts" / "audit_mcp_tools_resources.py"),
            "--timeout",
            str(float(mcp_timeout_s)),
            "--report-dir",
            str(report_dir),
        ]
        rc, out, err, dur = _run_cmd(mcp_cmd, env=env, timeout_s=timeout_s, cwd=root)
        mcp_path = _extract_report_path(out)
        if mcp_path is None or (not mcp_path.exists()):
            checks.append(
                CheckResult(
                    name="mcp_audit_matrix",
                    ok=False,
                    details=f"report_path_missing rc={rc} stderr={err.strip()[:260]}",
                    duration_ms=round(dur, 6),
                )
            )
            failures.append("mcp_audit_matrix")
        else:
            payload = _json_loads(mcp_path.read_text(encoding="utf-8"))
            if payload is None:
                checks.append(
                    CheckResult(
                        name="mcp_audit_matrix",
                        ok=False,
                        details=f"invalid report json at {mcp_path}",
                        duration_ms=round(dur, 6),
                    )
                )
                failures.append("mcp_audit_matrix")
            else:
                ok, details = evaluate_mcp_audit(payload, strict=strict)
                checks.append(
                    CheckResult(
                        name="mcp_audit_matrix",
                        ok=ok,
                        details=f"{details} report={mcp_path}",
                        duration_ms=round(dur, 6),
                    )
                )
                if not ok:
                    failures.append("mcp_audit_matrix")

    run_json_check(
        name="consciousness_status",
        cmd=[
            python_bin,
            str(root / "agent_forge" / "bin" / "eidctl"),
            "consciousness",
            "status",
            "--dir",
            str(state_dir),
            "--json",
        ],
        evaluator=evaluate_consciousness_status,
    )

    run_json_check(
        name="consciousness_benchmark",
        cmd=[
            python_bin,
            str(root / "agent_forge" / "bin" / "eidctl"),
            "consciousness",
            "benchmark",
            "--dir",
            str(state_dir),
            "--ticks",
            "3" if quick else "8",
            "--no-persist",
            "--json",
        ],
        evaluator=evaluate_benchmark,
    )

    run_json_check(
        name="consciousness_stress_benchmark",
        cmd=[
            python_bin,
            str(root / "agent_forge" / "bin" / "eidctl"),
            "consciousness",
            "stress-benchmark",
            "--dir",
            str(state_dir),
            "--ticks",
            "2" if quick else "4",
            "--event-fanout",
            "4" if quick else "6",
            "--broadcast-fanout",
            "2" if quick else "3",
            "--payload-chars",
            "4096",
            "--max-payload-bytes",
            "1024",
            "--no-persist",
            "--json",
        ],
        evaluator=lambda payload: evaluate_stress_benchmark(
            payload,
            strict_latency=bool(strict),
        ),
    )

    if not quick:
        run_json_check(
            name="consciousness_full_benchmark",
            cmd=[
                python_bin,
                str(root / "agent_forge" / "bin" / "eidctl"),
                "consciousness",
                "full-benchmark",
                "--dir",
                str(state_dir),
                "--rounds",
                "1",
                "--bench-ticks",
                "3",
                "--trial-ticks",
                "2",
                "--skip-llm",
                "--skip-red-team",
                "--json",
            ],
            evaluator=evaluate_full_benchmark,
        )

    report = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "root": str(root),
        "state_dir": str(state_dir),
        "python_bin": python_bin,
        "quick": bool(quick),
        "strict": bool(strict),
        "counts": {
            "checks_total": len(checks),
            "checks_ok": sum(1 for c in checks if c.ok),
            "checks_fail": sum(1 for c in checks if not c.ok),
        },
        "failed_checks": list(failures),
        "checks": [asdict(c) for c in checks],
    }
    path = report_dir / f"{run_id}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(path)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Linux parity audit matrix for Forge + MCP + consciousness runtime."
    )
    parser.add_argument("--python-bin", default="", help="Python interpreter to use for subprocess commands.")
    parser.add_argument("--state-dir", default="state", help="State directory for runtime checks.")
    parser.add_argument("--report-dir", default="reports", help="Directory to store audit report JSON.")
    parser.add_argument("--timeout", type=float, default=240.0, help="Per-command timeout seconds.")
    parser.add_argument("--mcp-timeout", type=float, default=10.0, help="Per-call timeout passed to MCP audit script.")
    parser.add_argument("--quick", action="store_true", help="Run reduced matrix (skips full-benchmark).")
    parser.add_argument("--strict", action="store_true", help="Fail when MCP soft fails are present.")
    parser.add_argument("--stdout-json", action="store_true", help="Print full report JSON to stdout.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python_bin = _python_bin(root, args.python_bin or None)
    report = run_matrix(
        root=root,
        python_bin=python_bin,
        state_dir=(
            (root / args.state_dir).resolve()
            if not Path(args.state_dir).is_absolute()
            else Path(args.state_dir).resolve()
        ),
        report_dir=(
            (root / args.report_dir).resolve()
            if not Path(args.report_dir).is_absolute()
            else Path(args.report_dir).resolve()
        ),
        timeout_s=max(10.0, float(args.timeout)),
        mcp_timeout_s=max(3.0, float(args.mcp_timeout)),
        quick=bool(args.quick),
        strict=bool(args.strict),
    )

    counts = report.get("counts") or {}
    print(f"Linux audit report: {report.get('report_path')}")
    print(
        "checks_total={checks_total} ok={checks_ok} fail={checks_fail}".format(
            checks_total=_safe_int(counts.get("checks_total")),
            checks_ok=_safe_int(counts.get("checks_ok")),
            checks_fail=_safe_int(counts.get("checks_fail")),
        )
    )
    failed = report.get("failed_checks") or []
    if failed:
        print(f"failed_checks={', '.join(str(x) for x in failed)}")
    if args.stdout_json:
        print(json.dumps(report, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
