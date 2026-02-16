from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "dependabot_remediation_plan.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("dependabot_remediation_plan", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("dependabot_remediation_plan", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


plan_mod = _load_module()


def _summary_payload() -> dict:
    return {
        "totals": {"alerts": 4, "open": 4, "fixed": 0},
        "open_high_critical": [
            {
                "number": 11,
                "severity": "critical",
                "ecosystem": "pip",
                "package": "urllib3",
                "manifest_path": "requirements.txt",
                "summary": "Critical vuln",
            },
            {
                "number": 12,
                "severity": "high",
                "ecosystem": "pip",
                "package": "requests",
                "manifest_path": "requirements.txt",
                "summary": "High vuln",
            },
            {
                "number": 13,
                "severity": "high",
                "ecosystem": "npm",
                "package": "glob",
                "manifest_path": "game_forge/src/autoseed/package-lock.json",
                "summary": "High vuln npm",
            },
            {
                "number": 14,
                "severity": "low",
                "ecosystem": "pip",
                "package": "pyyaml",
                "manifest_path": "requirements.txt",
                "summary": "Low vuln",
            },
        ],
    }


def test_build_remediation_plan_groups_and_scores() -> None:
    plan = plan_mod.build_remediation_plan(
        summary=_summary_payload(),
        repo="Ace1928/eidosian_forge",
        include_severities={"critical", "high"},
        max_batches=10,
    )
    assert plan["batch_count"] == 2
    assert plan["total_batched_alerts"] == 3
    first = plan["batches"][0]
    assert first["max_severity"] == "critical"
    assert first["manifest_path"] == "requirements.txt"
    assert first["severity_counts"]["critical"] == 1
    assert first["severity_counts"]["high"] == 1
    assert first["severity_counts"]["low"] == 0
    assert first["priority_score"] > 0


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary_payload()), encoding="utf-8")
    out_json = tmp_path / "plan.json"
    out_md = tmp_path / "plan.md"

    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--summary-json",
            str(summary_path),
            "--json-out",
            str(out_json),
            "--md-out",
            str(out_md),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert out_json.exists()
    assert out_md.exists()
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["batch_count"] >= 1
    md = out_md.read_text(encoding="utf-8")
    assert "# Dependabot Remediation Batches" in md
