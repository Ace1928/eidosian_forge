from __future__ import annotations

import importlib.machinery
import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "audit_workflow_action_pins.py"


def _load_module():
    loader = importlib.machinery.SourceFileLoader("audit_workflow_action_pins", str(SCRIPT_PATH))
    spec = importlib.util.spec_from_loader("audit_workflow_action_pins", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


audit = _load_module()


def test_scan_workflow_actions_ignores_local_and_docker(tmp_path: Path) -> None:
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    wf = wf_dir / "sample.yml"
    wf.write_text(
        "\n".join(
            [
                "name: sample",
                "jobs:",
                "  test:",
                "    steps:",
                "      - uses: actions/checkout@v4",
                "      - uses: actions/setup-python@v5",
                "      - uses: ./.github/actions/local",
                "      - uses: docker://alpine:3.19",
            ]
        ),
        encoding="utf-8",
    )

    rows = audit.scan_workflow_actions(wf_dir)
    specs = {str(row.get("spec") or "") for row in rows}
    assert "actions/checkout@v4" in specs
    assert "actions/setup-python@v5" in specs
    assert all(not spec.startswith("./") for spec in specs)
    assert all(not spec.startswith("docker://") for spec in specs)


def test_build_audit_report_detects_drift_unlock_and_mutable_ref(tmp_path: Path) -> None:
    wf_dir = tmp_path / ".github" / "workflows"
    wf_dir.mkdir(parents=True, exist_ok=True)
    (wf_dir / "sample.yml").write_text(
        "\n".join(
            [
                "name: sample",
                "jobs:",
                "  test:",
                "    steps:",
                "      - uses: actions/checkout@v4",
                "      - uses: actions/setup-node@main",
            ]
        ),
        encoding="utf-8",
    )
    usages = audit.scan_workflow_actions(wf_dir)

    def fake_resolver(owner: str, repo: str, ref: str) -> tuple[str, str]:
        _ = owner, repo
        if ref == "v4":
            return "1111111111111111111111111111111111111111", ""
        if ref == "main":
            return "2222222222222222222222222222222222222222", ""
        return "", "missing"

    lock = {
        "actions/checkout@v4": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    }
    report = audit.build_audit_report(usages, lock_pins=lock, resolver=fake_resolver)
    summary = report["summary"]

    assert summary["unique_action_refs"] == 2
    assert summary["drift_count"] == 1
    assert summary["unlocked_count"] == 1
    assert summary["mutable_ref_count"] == 1
