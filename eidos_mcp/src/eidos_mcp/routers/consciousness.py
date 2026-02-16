from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from contextlib import contextmanager
from typing import Optional

from eidosian_core import eidosian

from .. import FORGE_ROOT
from ..consciousness_protocol import ConsciousnessProtocol, DEFAULT_REFERENCES
from ..core import resource, tool
from ..forge_loader import ensure_forge_import

ensure_forge_import("agent_forge")

try:
    from agent_forge.consciousness import (
        ConsciousnessBenchmarkSuite,
        ConsciousnessKernel,
        ConsciousnessTrialRunner,
        IntegratedStackBenchmark,
    )
    from agent_forge.consciousness.perturb import Perturbation, make_drop, make_noise
except Exception:  # pragma: no cover - defensive for partial installs
    ConsciousnessBenchmarkSuite = None
    ConsciousnessKernel = None
    ConsciousnessTrialRunner = None
    IntegratedStackBenchmark = None
    Perturbation = None
    make_drop = None
    make_noise = None


def _protocol() -> ConsciousnessProtocol:
    return ConsciousnessProtocol()


def _state_dir(state_dir: Optional[str] = None) -> Path:
    if state_dir:
        return Path(state_dir).expanduser().resolve()
    return Path(
        os.environ.get("EIDOS_CONSCIOUSNESS_STATE_DIR", str(FORGE_ROOT / "state"))
    ).resolve()


def _runner(state_dir: Optional[str] = None) -> Optional[ConsciousnessTrialRunner]:
    if ConsciousnessTrialRunner is None:
        return None
    return ConsciousnessTrialRunner(_state_dir(state_dir))


def _bench(state_dir: Optional[str] = None) -> Optional[ConsciousnessBenchmarkSuite]:
    if ConsciousnessBenchmarkSuite is None:
        return None
    return ConsciousnessBenchmarkSuite(_state_dir(state_dir))


def _full_bench(state_dir: Optional[str] = None) -> Optional[IntegratedStackBenchmark]:
    if IntegratedStackBenchmark is None:
        return None
    return IntegratedStackBenchmark(_state_dir(state_dir))


@contextmanager
def _transient_state_dir(state_dir: Optional[str], persist: bool):
    if state_dir or persist:
        yield _state_dir(state_dir)
        return
    root = (_state_dir(None) / "_scratch").resolve()
    root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix="consciousness-bench-", dir=str(root))).resolve()
    try:
        yield scratch
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _bridge_status_payload(state_dir: Optional[str] = None) -> dict[str, object]:
    runner = _runner(state_dir)
    if runner is None:
        return {"error": "agent_forge consciousness runtime unavailable"}
    status = runner.status()
    return {
        "timestamp": status.get("timestamp"),
        "state_dir": status.get("state_dir"),
        "memory_bridge": status.get("memory_bridge") or {},
        "knowledge_bridge": status.get("knowledge_bridge") or {},
        "memory_recalls": status.get("memory_recalls"),
        "knowledge_hits": status.get("knowledge_hits"),
    }


@tool(
    name="consciousness_hypothesis_upsert",
    description="Create or update a falsifiable hypothesis used by the consciousness assessment protocol.",
    parameters={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "statement": {"type": "string"},
            "metric": {"type": "string"},
            "comparator": {"type": "string", "enum": [">=", "<=", ">", "<", "=="]},
            "threshold": {"type": "number"},
            "active": {"type": "boolean"},
            "source_url": {"type": "string"},
            "hypothesis_id": {"type": "string"},
        },
        "required": ["name", "statement", "metric", "comparator", "threshold"],
    },
)
@eidosian()
def consciousness_hypothesis_upsert(
    name: str,
    statement: str,
    metric: str,
    comparator: str,
    threshold: float,
    active: bool = True,
    source_url: str = "",
    hypothesis_id: Optional[str] = None,
) -> str:
    if comparator not in {">=", "<=", ">", "<", "=="}:
        return json.dumps(
            {
                "error": f"Unsupported comparator '{comparator}'",
                "supported": [">=", "<=", ">", "<", "=="],
            },
            indent=2,
        )

    protocol = _protocol()
    hypothesis = protocol.upsert_hypothesis(
        name=name,
        statement=statement,
        metric=metric,
        comparator=comparator,
        threshold=threshold,
        active=active,
        source_url=source_url,
        hypothesis_id=hypothesis_id,
    )
    return json.dumps(hypothesis, indent=2)


@tool(
    name="consciousness_hypothesis_list",
    description="List currently configured consciousness protocol hypotheses.",
    parameters={
        "type": "object",
        "properties": {"active_only": {"type": "boolean"}},
    },
)
@eidosian()
def consciousness_hypothesis_list(active_only: bool = False) -> str:
    protocol = _protocol()
    hypotheses = protocol.list_hypotheses(active_only=active_only)
    return json.dumps(hypotheses, indent=2)


@tool(
    name="consciousness_run_assessment",
    description="Run the operational consciousness assessment protocol and write a report.",
    parameters={
        "type": "object",
        "properties": {
            "trials": {"type": "integer"},
            "persist": {"type": "boolean"},
        },
    },
)
@eidosian()
def consciousness_run_assessment(trials: int = 3, persist: bool = True) -> str:
    protocol = _protocol()
    report = protocol.run_assessment(trials=trials, persist=persist)
    return json.dumps(report, indent=2)


@tool(
    name="consciousness_latest_report",
    description="Return the latest operational consciousness assessment report.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def consciousness_latest_report() -> str:
    protocol = _protocol()
    report = protocol.latest_report()
    if not report:
        return json.dumps({"error": "No report found"}, indent=2)
    return json.dumps(report, indent=2)


@tool(
    name="consciousness_get_report",
    description="Fetch a specific operational consciousness assessment report by report id.",
    parameters={
        "type": "object",
        "properties": {"report_id": {"type": "string"}},
        "required": ["report_id"],
    },
)
@eidosian()
def consciousness_get_report(report_id: str) -> str:
    protocol = _protocol()
    report = protocol.get_report(report_id)
    if not report:
        return json.dumps({"error": f"Report not found: {report_id}"}, indent=2)
    return json.dumps(report, indent=2)


@tool(
    name="consciousness_references",
    description="List primary references used to define assessment metrics and falsifiability.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def consciousness_references() -> str:
    return json.dumps(DEFAULT_REFERENCES, indent=2)


@tool(
    name="consciousness_kernel_status",
    description="Return runtime consciousness kernel status (workspace/coherence/rci/agency/boundary).",
    parameters={
        "type": "object",
        "properties": {
            "state_dir": {"type": "string"},
        },
    },
)
@eidosian()
def consciousness_kernel_status(state_dir: Optional[str] = None) -> str:
    runner = _runner(state_dir)
    if runner is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    return json.dumps(runner.status(), indent=2)


@tool(
    name="consciousness_bridge_status",
    description="Return memory_forge and knowledge_forge bridge status from the runtime consciousness stack.",
    parameters={
        "type": "object",
        "properties": {
            "state_dir": {"type": "string"},
        },
    },
)
@eidosian()
def consciousness_bridge_status(state_dir: Optional[str] = None) -> str:
    return json.dumps(_bridge_status_payload(state_dir), indent=2)


@tool(
    name="consciousness_kernel_trial",
    description="Run a consciousness perturbation trial against the runtime kernel and return report.",
    parameters={
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": ["noise", "drop", "zero", "jitter"]},
            "target": {"type": "string"},
            "magnitude": {"type": "number"},
            "duration_s": {"type": "number"},
            "ticks": {"type": "integer"},
            "persist": {"type": "boolean"},
            "state_dir": {"type": "string"},
        },
    },
)
@eidosian()
def consciousness_kernel_trial(
    kind: str = "noise",
    target: str = "attention",
    magnitude: float = 0.2,
    duration_s: float = 1.0,
    ticks: int = 3,
    persist: bool = True,
    state_dir: Optional[str] = None,
) -> str:
    runner = _runner(state_dir)
    if runner is None or ConsciousnessKernel is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)

    if kind == "noise" and make_noise is not None:
        perturbation = make_noise(target, magnitude, duration_s)
    elif kind == "drop" and make_drop is not None:
        perturbation = make_drop(target, duration_s)
    elif Perturbation is not None:
        perturbation = Perturbation(
            kind=kind,
            target=target,
            magnitude=float(magnitude),
            duration_s=float(duration_s),
            meta={},
        )
    else:
        return json.dumps({"error": "perturbation runtime unavailable"}, indent=2)

    kernel = ConsciousnessKernel(_state_dir(state_dir))
    result = runner.run_trial(
        kernel=kernel,
        perturbation=perturbation,
        ticks=max(1, int(ticks)),
        persist=bool(persist),
    )
    return json.dumps(result.report, indent=2)


@tool(
    name="consciousness_kernel_latest_trial",
    description="Return latest persisted consciousness kernel trial report.",
    parameters={
        "type": "object",
        "properties": {
            "state_dir": {"type": "string"},
        },
    },
)
@eidosian()
def consciousness_kernel_latest_trial(state_dir: Optional[str] = None) -> str:
    runner = _runner(state_dir)
    if runner is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    latest = runner.latest_trial()
    if latest is None:
        return json.dumps({"error": "No trial report found"}, indent=2)
    return json.dumps(latest, indent=2)


@tool(
    name="consciousness_kernel_benchmark",
    description="Run internal consciousness benchmark suite and return scored results.",
    parameters={
        "type": "object",
        "properties": {
            "ticks": {"type": "integer"},
            "persist": {"type": "boolean"},
            "state_dir": {"type": "string"},
            "external_scores": {"type": "object"},
            "external_sources": {"type": "object"},
        },
    },
)
@eidosian()
def consciousness_kernel_benchmark(
    ticks: int = 12,
    persist: bool = True,
    state_dir: Optional[str] = None,
    external_scores: Optional[dict[str, float]] = None,
    external_sources: Optional[dict[str, str]] = None,
) -> str:
    with _transient_state_dir(state_dir, bool(persist)) as run_state_dir:
        bench = _bench(str(run_state_dir))
        if bench is None or ConsciousnessKernel is None:
            return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
        kernel = ConsciousnessKernel(run_state_dir)
        result = bench.run(
            kernel=kernel,
            ticks=max(1, int(ticks)),
            persist=bool(persist),
            external_scores=external_scores or {},
            external_sources=external_sources or {},
        )
        return json.dumps(result.report, indent=2)


@tool(
    name="consciousness_kernel_latest_benchmark",
    description="Return latest persisted consciousness benchmark report.",
    parameters={
        "type": "object",
        "properties": {
            "state_dir": {"type": "string"},
        },
    },
)
@eidosian()
def consciousness_kernel_latest_benchmark(state_dir: Optional[str] = None) -> str:
    bench = _bench(state_dir)
    if bench is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    latest = bench.latest_benchmark()
    if latest is None:
        return json.dumps({"error": "No benchmark report found"}, indent=2)
    return json.dumps(latest, indent=2)


@tool(
    name="consciousness_kernel_full_benchmark",
    description="Run integrated consciousness benchmark (core/trials + optional MCP + optional local LLM).",
    parameters={
        "type": "object",
        "properties": {
            "rounds": {"type": "integer"},
            "bench_ticks": {"type": "integer"},
            "trial_ticks": {"type": "integer"},
            "run_mcp": {"type": "boolean"},
            "run_llm": {"type": "boolean"},
            "persist": {"type": "boolean"},
            "state_dir": {"type": "string"},
            "llm_model": {"type": "string"},
            "ollama_endpoint": {"type": "string"},
            "timeout_sec": {"type": "number"},
        },
    },
)
@eidosian()
def consciousness_kernel_full_benchmark(
    rounds: int = 2,
    bench_ticks: int = 6,
    trial_ticks: int = 2,
    run_mcp: bool = False,
    run_llm: bool = False,
    persist: bool = True,
    state_dir: Optional[str] = None,
    llm_model: str = "qwen2.5:1.5b",
    ollama_endpoint: str = "http://127.0.0.1:11434",
    timeout_sec: float = 45.0,
) -> str:
    with _transient_state_dir(state_dir, bool(persist)) as run_state_dir:
        full = _full_bench(str(run_state_dir))
        if full is None:
            return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
        result = full.run(
            rounds=max(1, int(rounds)),
            bench_ticks=max(1, int(bench_ticks)),
            trial_ticks=max(1, int(trial_ticks)),
            run_mcp=bool(run_mcp),
            run_llm=bool(run_llm),
            persist=bool(persist),
            llm_model=llm_model,
            ollama_endpoint=ollama_endpoint,
            timeout_sec=max(1.0, float(timeout_sec)),
        )
        return json.dumps(result.report, indent=2)


@tool(
    name="consciousness_kernel_latest_full_benchmark",
    description="Return latest persisted integrated consciousness benchmark report.",
    parameters={
        "type": "object",
        "properties": {
            "state_dir": {"type": "string"},
        },
    },
)
@eidosian()
def consciousness_kernel_latest_full_benchmark(state_dir: Optional[str] = None) -> str:
    full = _full_bench(state_dir)
    if full is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    latest = full.latest()
    if latest is None:
        return json.dumps({"error": "No integrated benchmark report found"}, indent=2)
    return json.dumps(latest, indent=2)


@resource(
    uri="eidos://consciousness/hypotheses",
    description="Configured falsifiable hypotheses for the consciousness assessment protocol.",
)
@eidosian()
def consciousness_hypotheses_resource() -> str:
    protocol = _protocol()
    return json.dumps(protocol.list_hypotheses(active_only=False), indent=2)


@resource(
    uri="eidos://consciousness/latest",
    description="Latest operational consciousness assessment report.",
)
@eidosian()
def consciousness_latest_resource() -> str:
    protocol = _protocol()
    report = protocol.latest_report()
    if not report:
        return json.dumps({"error": "No report found"}, indent=2)
    return json.dumps(report, indent=2)


@resource(
    uri="eidos://consciousness/runtime-status",
    description="Runtime consciousness kernel status snapshot.",
)
@eidosian()
def consciousness_runtime_status_resource() -> str:
    runner = _runner()
    if runner is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    return json.dumps(runner.status(), indent=2)


@resource(
    uri="eidos://consciousness/runtime-integrations",
    description="Runtime integration status for memory_bridge and knowledge_bridge modules.",
)
@eidosian()
def consciousness_runtime_integrations_resource() -> str:
    return json.dumps(_bridge_status_payload(), indent=2)


@resource(
    uri="eidos://consciousness/runtime-latest-trial",
    description="Latest runtime consciousness perturbation trial report.",
)
@eidosian()
def consciousness_runtime_latest_trial_resource() -> str:
    runner = _runner()
    if runner is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    latest = runner.latest_trial()
    if latest is None:
        return json.dumps({"error": "No trial report found"}, indent=2)
    return json.dumps(latest, indent=2)


@resource(
    uri="eidos://consciousness/runtime-latest-benchmark",
    description="Latest runtime consciousness benchmark report.",
)
@eidosian()
def consciousness_runtime_latest_benchmark_resource() -> str:
    bench = _bench()
    if bench is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    latest = bench.latest_benchmark()
    if latest is None:
        return json.dumps({"error": "No benchmark report found"}, indent=2)
    return json.dumps(latest, indent=2)


@resource(
    uri="eidos://consciousness/runtime-latest-full-benchmark",
    description="Latest runtime integrated consciousness benchmark report.",
)
@eidosian()
def consciousness_runtime_latest_full_benchmark_resource() -> str:
    full = _full_bench()
    if full is None:
        return json.dumps({"error": "agent_forge consciousness runtime unavailable"}, indent=2)
    latest = full.latest()
    if latest is None:
        return json.dumps({"error": "No integrated benchmark report found"}, indent=2)
    return json.dumps(latest, indent=2)
