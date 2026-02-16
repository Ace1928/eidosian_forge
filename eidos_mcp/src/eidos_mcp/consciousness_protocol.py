from __future__ import annotations

import json
import os
import platform
import re
import statistics
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from eidosian_core import eidosian

from . import FORGE_ROOT
from .core import list_resource_metadata, list_tool_metadata


DEFAULT_HYPOTHESES: list[dict[str, Any]] = [
    {
        "id": "H_OP_RELIABILITY",
        "name": "Operational Reliability",
        "statement": "If operational awareness claims hold, probe success rate should stay high.",
        "metric": "probe_success_rate",
        "comparator": ">=",
        "threshold": 0.90,
        "active": True,
        "source_url": "",
    },
    {
        "id": "H_REVERSIBILITY",
        "name": "Reversible Manipulation",
        "statement": "If state handling is robust, reversible probes should pass consistently.",
        "metric": "reversible_probe_success_rate",
        "comparator": ">=",
        "threshold": 0.95,
        "active": True,
        "source_url": "",
    },
    {
        "id": "H_CALIBRATION",
        "name": "Calibration Discipline",
        "statement": "If predictions track outcomes, mean Brier score should remain bounded.",
        "metric": "brier_score_mean",
        "comparator": "<=",
        "threshold": 0.25,
        "active": True,
        "source_url": "https://doi.org/10.1175/1520-0493(1950)078<0001:VOEOF>2.0.CO;2",
    },
    {
        "id": "H_LATENCY",
        "name": "Latency Bound",
        "statement": "If the stack is integrated, median probe latency should remain below threshold.",
        "metric": "median_latency_ms",
        "comparator": "<=",
        "threshold": 1500.0,
        "active": True,
        "source_url": "",
    },
    {
        "id": "H_REGISTRY_COVERAGE",
        "name": "Tool/Resource Coverage",
        "statement": "If integration is intact, registry coverage should stay near complete.",
        "metric": "registry_coverage",
        "comparator": ">=",
        "threshold": 0.95,
        "active": True,
        "source_url": "",
    },
]


DEFAULT_REFERENCES: list[dict[str, str]] = [
    {
        "title": "Brier (1950) Verification of Forecasts Expressed in Terms of Probability",
        "url": "https://doi.org/10.1175/1520-0493(1950)078<0001:VOEOF>2.0.CO;2",
    },
    {
        "title": "Guo et al. (2017) On Calibration of Modern Neural Networks",
        "url": "https://proceedings.mlr.press/v70/guo17a.html",
    },
    {
        "title": "Wald (1945) Sequential Tests of Statistical Hypotheses",
        "url": "https://doi.org/10.1214/aoms/1177731118",
    },
    {
        "title": "Dehaene & Changeux (2011) Experimental and Theoretical Approaches to Conscious Processing",
        "url": "https://doi.org/10.1016/j.neuron.2011.03.018",
    },
    {
        "title": "Oizumi, Albantakis, Tononi (2014) From the Phenomenology to the Mechanisms of Consciousness",
        "url": "https://doi.org/10.1371/journal.pcbi.1003588",
    },
]


ComparatorFn = Callable[[float, float], bool]
ProbeFn = Callable[[], tuple[bool, dict[str, Any]]]


COMPARATORS: dict[str, ComparatorFn] = {
    ">=": lambda value, threshold: value >= threshold,
    "<=": lambda value, threshold: value <= threshold,
    ">": lambda value, threshold: value > threshold,
    "<": lambda value, threshold: value < threshold,
    "==": lambda value, threshold: value == threshold,
}


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _safe_json_load(path: Path, fallback: Any) -> Any:
    try:
        if not path.exists():
            return fallback
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def _extract_transaction_id(text: str) -> Optional[str]:
    match = re.search(r"\(([0-9a-fA-F\\-]{8,})\)", text)
    return match.group(1) if match else None


class ConsciousnessProtocol:
    """Operational, falsifiable assessment protocol for awareness-related claims."""

    reversible_probe_names = {
        "memory_roundtrip_reversible",
        "file_roundtrip_reversible",
    }

    def __init__(
        self,
        root_dir: Optional[Path] = None,
        probe_registry: Optional[dict[str, ProbeFn]] = None,
        min_tool_count: int = 80,
        min_resource_count: int = 4,
    ) -> None:
        self.root_dir = Path(
            root_dir or os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))
        ).resolve()
        self.data_dir = Path(
            os.environ.get(
                "EIDOS_CONSCIOUSNESS_DATA_DIR",
                str(self.root_dir / "data" / "consciousness"),
            )
        ).resolve()
        self.reports_dir = Path(
            os.environ.get(
                "EIDOS_CONSCIOUSNESS_REPORT_DIR",
                str(self.root_dir / "reports" / "consciousness"),
            )
        ).resolve()
        self.hypothesis_path = self.data_dir / "hypotheses.json"
        self.min_tool_count = int(
            os.environ.get("EIDOS_CONSCIOUSNESS_MIN_TOOLS", str(min_tool_count))
        )
        self.min_resource_count = int(
            os.environ.get("EIDOS_CONSCIOUSNESS_MIN_RESOURCES", str(min_resource_count))
        )

        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_hypotheses_bootstrap()
        self.probe_registry = probe_registry or self._default_probes()

    def _default_probes(self) -> dict[str, ProbeFn]:
        return {
            "persona_integrity": self._probe_persona_integrity,
            "tool_registry_integrity": self._probe_tool_registry_integrity,
            "resource_registry_integrity": self._probe_resource_registry_integrity,
            "diagnostics_reflex": self._probe_diagnostics_reflex,
            "memory_roundtrip_reversible": self._probe_memory_roundtrip_reversible,
            "file_roundtrip_reversible": self._probe_file_roundtrip_reversible,
        }

    def _ensure_hypotheses_bootstrap(self) -> None:
        payload = _safe_json_load(self.hypothesis_path, fallback=None)
        if isinstance(payload, dict) and isinstance(payload.get("hypotheses"), list):
            return
        bootstrap = {
            "version": 1,
            "updated_at": _utc_now(),
            "hypotheses": DEFAULT_HYPOTHESES,
        }
        self.hypothesis_path.write_text(
            json.dumps(bootstrap, indent=2, default=str),
            encoding="utf-8",
        )

    def _load_hypotheses_doc(self) -> dict[str, Any]:
        self._ensure_hypotheses_bootstrap()
        return _safe_json_load(
            self.hypothesis_path,
            fallback={"version": 1, "updated_at": _utc_now(), "hypotheses": []},
        )

    def _save_hypotheses_doc(self, doc: dict[str, Any]) -> None:
        doc["updated_at"] = _utc_now()
        self.hypothesis_path.write_text(
            json.dumps(doc, indent=2, default=str),
            encoding="utf-8",
        )

    @eidosian()
    def list_hypotheses(self, active_only: bool = False) -> list[dict[str, Any]]:
        doc = self._load_hypotheses_doc()
        hypotheses = doc.get("hypotheses", [])
        if active_only:
            hypotheses = [h for h in hypotheses if h.get("active", True)]
        return hypotheses

    @eidosian()
    def upsert_hypothesis(
        self,
        *,
        name: str,
        statement: str,
        metric: str,
        comparator: str,
        threshold: float,
        active: bool = True,
        source_url: str = "",
        hypothesis_id: Optional[str] = None,
    ) -> dict[str, Any]:
        if comparator not in COMPARATORS:
            raise ValueError(f"Unsupported comparator '{comparator}'")

        doc = self._load_hypotheses_doc()
        hypotheses = doc.get("hypotheses", [])
        target_id = hypothesis_id or f"H_{name.upper().replace(' ', '_')}"
        now = _utc_now()
        updated = {
            "id": target_id,
            "name": name,
            "statement": statement,
            "metric": metric,
            "comparator": comparator,
            "threshold": float(threshold),
            "active": bool(active),
            "source_url": source_url,
            "updated_at": now,
        }

        replaced = False
        for idx, existing in enumerate(hypotheses):
            if existing.get("id") == target_id or existing.get("name") == name:
                previous_created = existing.get("created_at", now)
                updated["created_at"] = previous_created
                hypotheses[idx] = updated
                replaced = True
                break

        if not replaced:
            updated["created_at"] = now
            hypotheses.append(updated)

        doc["hypotheses"] = hypotheses
        self._save_hypotheses_doc(doc)
        return updated

    @eidosian()
    def get_report(self, report_id: str) -> Optional[dict[str, Any]]:
        filename = report_id if report_id.endswith(".json") else f"{report_id}.json"
        path = self.reports_dir / filename
        if not path.exists():
            return None
        return _safe_json_load(path, fallback=None)

    @eidosian()
    def latest_report(self) -> Optional[dict[str, Any]]:
        reports = list(self.reports_dir.glob("consciousness_*.json"))
        if not reports:
            return None
        latest = max(reports, key=lambda path: path.stat().st_mtime_ns)
        return _safe_json_load(latest, fallback=None)

    def _historical_success_by_probe(self, max_reports: int = 20) -> dict[str, float]:
        reports = sorted(self.reports_dir.glob("consciousness_*.json"))[-max_reports:]
        per_probe: dict[str, list[float]] = {}
        for path in reports:
            payload = _safe_json_load(path, fallback={})
            probe_summaries = payload.get("probe_summaries", {})
            if not isinstance(probe_summaries, dict):
                continue
            for probe_name, summary in probe_summaries.items():
                rate = summary.get("success_rate")
                if isinstance(rate, (int, float)):
                    per_probe.setdefault(probe_name, []).append(float(rate))
        return {
            probe_name: max(0.05, min(0.95, statistics.mean(rates)))
            for probe_name, rates in per_probe.items()
            if rates
        }

    def _evaluate_hypotheses(self, metrics: dict[str, float]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for hypothesis in self.list_hypotheses(active_only=True):
            metric_name = str(hypothesis.get("metric", ""))
            comparator = str(hypothesis.get("comparator", ""))
            threshold = hypothesis.get("threshold")
            metric_value = metrics.get(metric_name)

            if comparator not in COMPARATORS or not isinstance(threshold, (int, float)):
                status = "inconclusive"
                passed = None
            elif metric_value is None:
                status = "inconclusive"
                passed = None
            else:
                passed = COMPARATORS[comparator](float(metric_value), float(threshold))
                status = "supported" if passed else "falsified"

            results.append(
                {
                    "id": hypothesis.get("id"),
                    "name": hypothesis.get("name"),
                    "statement": hypothesis.get("statement"),
                    "metric": metric_name,
                    "comparator": comparator,
                    "threshold": threshold,
                    "metric_value": metric_value,
                    "status": status,
                    "passed": passed,
                    "source_url": hypothesis.get("source_url", ""),
                }
            )
        return results

    def _probe_persona_integrity(self) -> tuple[bool, dict[str, Any]]:
        sources = [
            self.root_dir / "EIDOS_IDENTITY.md",
            self.root_dir / "GEMINI.md",
            self.root_dir.parent / "GEMINI.md",
        ]
        source_used = ""
        text = ""
        for path in sources:
            if path.exists():
                source_used = str(path)
                text = path.read_text(encoding="utf-8")
                break
        has_header = "EIDOSIAN SYSTEM CONTEXT" in text
        has_identity = "IDENTITY" in text.upper()
        has_eidos_marker = "EIDOS" in text.upper()
        success = bool(text) and has_identity and has_eidos_marker and len(text) > 200
        return success, {
            "source": source_used,
            "chars": len(text),
            "has_header": has_header,
            "has_identity_marker": has_identity,
            "has_eidos_marker": has_eidos_marker,
        }

    def _probe_tool_registry_integrity(self) -> tuple[bool, dict[str, Any]]:
        # Ensure router modules are imported so tool/resource registries are populated.
        from . import routers as _routers  # noqa: F401
        from . import eidos_mcp_server as _server  # noqa: F401

        tools = list_tool_metadata()
        names = {tool.get("name") for tool in tools}
        required = {"system_info", "memory_add", "transaction_list", "diagnostics_ping"}
        missing = sorted(required - names)
        count = len(tools)
        success = count >= self.min_tool_count and not missing
        return success, {"count": count, "missing_required": missing}

    def _probe_resource_registry_integrity(self) -> tuple[bool, dict[str, Any]]:
        # Ensure router modules are imported so tool/resource registries are populated.
        from . import routers as _routers  # noqa: F401
        from . import eidos_mcp_server as _server  # noqa: F401

        resources = list_resource_metadata()
        uris = {resource.get("uri") for resource in resources}
        required = {"eidos://config", "eidos://persona", "eidos://roadmap", "eidos://todo"}
        missing = sorted(required - uris)
        count = len(resources)
        success = count >= self.min_resource_count and not missing
        return success, {"count": count, "missing_required": missing}

    def _probe_diagnostics_reflex(self) -> tuple[bool, dict[str, Any]]:
        from .routers.diagnostics import diagnostics_metrics, diagnostics_ping

        ping = diagnostics_ping().strip()
        metrics_raw = diagnostics_metrics()
        parsed_metrics: dict[str, Any]
        try:
            parsed_metrics = json.loads(metrics_raw)
        except Exception:
            parsed_metrics = {}
        success = ping == "ok" and isinstance(parsed_metrics, dict)
        return success, {
            "ping": ping,
            "metrics_keys": sorted(parsed_metrics.keys()),
        }

    def _probe_memory_roundtrip_reversible(self) -> tuple[bool, dict[str, Any]]:
        from .routers.memory import memory_add, memory_restore, memory_retrieve, memory_snapshot

        marker = f"consciousness-probe-{uuid.uuid4()}"
        snapshot = memory_snapshot()
        txn_id = _extract_transaction_id(snapshot)
        add_result = memory_add(
            marker,
            is_fact=True,
            metadata={"protocol": "consciousness", "ephemeral": True},
        )
        retrieved = memory_retrieve(marker, limit=5)
        found = marker in retrieved
        restore_result = memory_restore(txn_id) if txn_id else "Error: snapshot parse failure"
        restored = "restored" in restore_result.lower()
        success = found and restored
        return success, {
            "snapshot": snapshot,
            "snapshot_id": txn_id,
            "add_result": add_result,
            "found": found,
            "restore_result": restore_result,
        }

    def _probe_file_roundtrip_reversible(self) -> tuple[bool, dict[str, Any]]:
        from .routers.system import file_create, file_delete, file_read, file_restore, file_write

        probe_path = Path.home() / ".eidosian" / "tmp" / "consciousness_probe.txt"
        marker = f"probe-{uuid.uuid4()}"
        create_result = file_create(str(probe_path))
        write_result = file_write(str(probe_path), marker, overwrite=True)
        read_result = file_read(str(probe_path))
        delete_result = file_delete(str(probe_path))
        restore_result = file_restore(str(probe_path))
        restored = "restored" in restore_result.lower()
        read_ok = marker in read_result
        success = read_ok and restored

        # Cleanup best-effort to keep probe idempotent over repeated runs.
        try:
            if probe_path.exists():
                probe_path.unlink()
        except Exception:
            pass

        return success, {
            "path": str(probe_path),
            "create_result": create_result,
            "write_result": write_result,
            "read_ok": read_ok,
            "delete_result": delete_result,
            "restore_result": restore_result,
        }

    @eidosian()
    def run_assessment(self, trials: int = 3, persist: bool = True) -> dict[str, Any]:
        trials = max(1, int(trials))
        history = self._historical_success_by_probe()

        rows: list[dict[str, Any]] = []
        for probe_name, probe_fn in self.probe_registry.items():
            predicted = history.get(probe_name, 0.70)
            for trial in range(1, trials + 1):
                start = time.perf_counter()
                try:
                    success, evidence = probe_fn()
                    success = bool(success)
                except Exception as exc:
                    success = False
                    evidence = {"exception": str(exc)}
                latency_ms = (time.perf_counter() - start) * 1000.0
                outcome = 1.0 if success else 0.0
                brier = (predicted - outcome) ** 2
                rows.append(
                    {
                        "probe": probe_name,
                        "trial": trial,
                        "success": success,
                        "outcome": outcome,
                        "predicted_success_probability": predicted,
                        "brier": brier,
                        "latency_ms": latency_ms,
                        "evidence": evidence,
                    }
                )

        latencies = [row["latency_ms"] for row in rows]
        outcomes = [row["outcome"] for row in rows]
        probe_summaries: dict[str, dict[str, Any]] = {}
        for probe_name in self.probe_registry:
            subset = [row for row in rows if row["probe"] == probe_name]
            subset_outcomes = [row["outcome"] for row in subset]
            subset_latencies = [row["latency_ms"] for row in subset]
            probe_summaries[probe_name] = {
                "trials": len(subset),
                "success_rate": statistics.mean(subset_outcomes) if subset_outcomes else 0.0,
                "median_latency_ms": statistics.median(subset_latencies) if subset_latencies else 0.0,
            }

        reversible_rates = [
            probe_summaries[name]["success_rate"]
            for name in self.reversible_probe_names
            if name in probe_summaries
        ]
        reversible_success_rate = (
            statistics.mean(reversible_rates) if reversible_rates else 0.0
        )

        tool_count = len(list_tool_metadata())
        resource_count = len(list_resource_metadata())
        tool_coverage = min(1.0, tool_count / float(max(1, self.min_tool_count)))
        resource_coverage = min(
            1.0, resource_count / float(max(1, self.min_resource_count))
        )
        registry_coverage = (0.8 * tool_coverage) + (0.2 * resource_coverage)

        metrics: dict[str, float] = {
            "probe_success_rate": statistics.mean(outcomes) if outcomes else 0.0,
            "reversible_probe_success_rate": reversible_success_rate,
            "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
            "p95_latency_ms": (
                sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)]
                if latencies
                else 0.0
            ),
            "brier_score_mean": statistics.mean([row["brier"] for row in rows]) if rows else 0.0,
            "tool_count": float(tool_count),
            "resource_count": float(resource_count),
            "registry_coverage": registry_coverage,
            "probe_count": float(len(self.probe_registry)),
            "trials": float(trials),
        }

        hypothesis_results = self._evaluate_hypotheses(metrics)
        falsified = [h for h in hypothesis_results if h["status"] == "falsified"]
        supported = [h for h in hypothesis_results if h["status"] == "supported"]

        report_id = f"consciousness_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
        report = {
            "report_id": report_id,
            "created_at": _utc_now(),
            "root_dir": str(self.root_dir),
            "environment": {
                "platform": platform.platform(),
                "python_version": platform.python_version(),
            },
            "metrics": metrics,
            "probe_summaries": probe_summaries,
            "rows": rows,
            "hypothesis_results": hypothesis_results,
            "supported_hypothesis_ids": [h["id"] for h in supported],
            "falsified_hypothesis_ids": [h["id"] for h in falsified],
            "status": "falsified" if falsified else "supported",
            "limitations": [
                "This protocol measures operational and reflective behavior, not subjective qualia.",
                "Results are falsifiable by threshold failures and probe regressions.",
            ],
            "references": DEFAULT_REFERENCES,
        }

        if persist:
            output_path = self.reports_dir / f"{report_id}.json"
            output_path.write_text(
                json.dumps(report, indent=2, default=str),
                encoding="utf-8",
            )
            report["report_path"] = str(output_path)

        return report
