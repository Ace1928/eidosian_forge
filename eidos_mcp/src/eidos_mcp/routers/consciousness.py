from __future__ import annotations

import json
from typing import Optional

from eidosian_core import eidosian

from ..consciousness_protocol import ConsciousnessProtocol, DEFAULT_REFERENCES
from ..core import resource, tool


def _protocol() -> ConsciousnessProtocol:
    return ConsciousnessProtocol()


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
