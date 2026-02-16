from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "sync_security_remediation_issues.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("sync_security_remediation_issues", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("sync_security_remediation_issues", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


sync_mod = _load_module()


def _batch(key: str, severity: str, manifest: str) -> dict:
    return {
        "batch_key": key,
        "ecosystem": "pip",
        "manifest_path": manifest,
        "max_severity": severity,
        "total_alerts": 2,
        "top_packages": [{"name": "urllib3", "count": 1}],
        "alerts": [{"number": 10, "severity": severity, "package": "urllib3", "summary": "x"}],
        "overflow_alert_count": 0,
        "suggested_branch": "security/remediation/test",
        "suggested_pr_title": "security: remediate",
    }


def test_extract_batch_key_and_render_body_marker() -> None:
    batch = _batch("pip::requirements-txt", "high", "requirements.txt")
    body = sync_mod._render_issue_body(batch)
    key = sync_mod._extract_batch_key(body)
    assert key == "pip::requirements-txt"


def test_sync_remediation_issues_dry_run_actions(monkeypatch) -> None:
    existing_body = (
        "<!-- eidos-security-remediation-batch:pip::requirements-txt -->\n"
        "# Security Remediation Batch\n"
    )
    existing = [
        {"number": 101, "title": "old title", "body": existing_body},
        {"number": 102, "title": "dup title", "body": existing_body},
        {
            "number": 103,
            "title": "stale",
            "body": "<!-- eidos-security-remediation-batch:npm::lockfile -->",
        },
    ]
    monkeypatch.setattr(sync_mod, "_list_open_issues", lambda *_args, **_kwargs: existing)

    plan = {
        "batches": [
            _batch("pip::requirements-txt", "critical", "requirements.txt"),
            _batch("pip::pyproject-toml", "high", "pyproject.toml"),
        ]
    }
    report = sync_mod.sync_remediation_issues(
        repo="Ace1928/eidosian_forge",
        token="x",
        plan=plan,
        dry_run=True,
    )
    assert report["created_count"] == 1
    assert report["updated_count"] == 1
    assert report["closed_stale_count"] == 1
    assert report["closed_duplicate_count"] == 1
