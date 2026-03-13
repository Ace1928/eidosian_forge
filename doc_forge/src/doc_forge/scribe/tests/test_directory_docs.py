from __future__ import annotations

import json
import subprocess
from pathlib import Path

from doc_forge.scribe.directory_docs import (
    inventory_summary,
    inventory_status,
    load_policy,
    readme_diff,
    render_directory_readme,
    upsert_directory_readme,
)


def _git(args: list[str], cwd: Path) -> None:
    subprocess.run(["git", *args], cwd=str(cwd), check=True, capture_output=True, text=True)


def test_inventory_summary_detects_missing_readme(temp_forge_root: Path) -> None:
    _git(["init"], temp_forge_root)
    _git(["config", "user.email", "test@example.com"], temp_forge_root)
    _git(["config", "user.name", "Test"], temp_forge_root)
    tracked = temp_forge_root / "doc_forge" / "src" / "doc_forge" / "scribe"
    tracked.mkdir(parents=True, exist_ok=True)
    (tracked / "service.py").write_text("from fastapi import FastAPI\napp = FastAPI()\n", encoding="utf-8")
    policy_dir = temp_forge_root / "cfg"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "documentation_policy.json").write_text(
        json.dumps(
            {
                "documented_prefixes": ["doc_forge"],
                "excluded_prefixes": [],
                "excluded_segments": [],
            }
        ),
        encoding="utf-8",
    )
    _git(["add", "."], temp_forge_root)
    _git(["commit", "-m", "init"], temp_forge_root)

    payload = inventory_summary(temp_forge_root)
    assert payload["required_directory_count"] >= 1
    assert "doc_forge/src/doc_forge/scribe" in payload["missing_readmes"]


def test_render_diff_and_upsert(temp_forge_root: Path) -> None:
    _git(["init"], temp_forge_root)
    _git(["config", "user.email", "test@example.com"], temp_forge_root)
    _git(["config", "user.name", "Test"], temp_forge_root)
    target = temp_forge_root / "web_interface_forge" / "src" / "web_interface_forge" / "dashboard"
    target.mkdir(parents=True, exist_ok=True)
    (target / "main.py").write_text(
        'from fastapi import FastAPI\napp = FastAPI()\n@app.get("/health")\ndef health():\n    return {"ok": True}\n',
        encoding="utf-8",
    )
    policy_dir = temp_forge_root / "cfg"
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / "documentation_policy.json").write_text(
        json.dumps(
            {
                "documented_prefixes": ["web_interface_forge"],
                "excluded_prefixes": [],
                "excluded_segments": [],
            }
        ),
        encoding="utf-8",
    )
    _git(["add", "."], temp_forge_root)
    _git(["commit", "-m", "init"], temp_forge_root)

    content = render_directory_readme(temp_forge_root, "web_interface_forge/src/web_interface_forge/dashboard")
    assert "GET /health" in content
    diff = readme_diff(temp_forge_root, "web_interface_forge/src/web_interface_forge/dashboard")
    assert "README.md" in diff
    result = upsert_directory_readme(temp_forge_root, "web_interface_forge/src/web_interface_forge/dashboard")
    assert result["changed"] is True
    assert (target / "README.md").exists()
    assert "Accuracy Contract" in (target / "README.md").read_text(encoding="utf-8")


def test_render_includes_parent_readme_reference_and_router_routes(temp_forge_root: Path) -> None:
    _git(["init"], temp_forge_root)
    _git(["config", "user.email", "test@example.com"], temp_forge_root)
    _git(["config", "user.name", "Test"], temp_forge_root)
    parent = temp_forge_root / "doc_forge" / "src" / "doc_forge"
    target = parent / "scribe"
    target.mkdir(parents=True, exist_ok=True)
    (parent / "README.md").write_text("# Parent\n", encoding="utf-8")
    (target / "service.py").write_text(
        'from fastapi import APIRouter\nrouter = APIRouter()\n@router.get("/coverage")\ndef coverage():\n    return {"ok": True}\n',
        encoding="utf-8",
    )
    cfg_dir = temp_forge_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"documented_prefixes": ["doc_forge"], "excluded_prefixes": [], "excluded_segments": []}),
        encoding="utf-8",
    )
    _git(["add", "."], temp_forge_root)
    _git(["commit", "-m", "init"], temp_forge_root)

    content = render_directory_readme(temp_forge_root, "doc_forge/src/doc_forge/scribe")
    assert "GET /coverage" in content
    assert "Parent README" in content
    assert "[`doc_forge/src/doc_forge/README.md`](../README.md)" in content


def test_inventory_status_and_test_reference_detection(temp_forge_root: Path) -> None:
    _git(["init"], temp_forge_root)
    _git(["config", "user.email", "test@example.com"], temp_forge_root)
    _git(["config", "user.name", "Test"], temp_forge_root)
    target = temp_forge_root / "code_forge" / "src" / "code_forge" / "library"
    tests = temp_forge_root / "code_forge" / "tests"
    target.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    (target / "db.py").write_text("def x():\n    return 1\n", encoding="utf-8")
    (tests / "test_library_db.py").write_text("def test_x():\n    assert True\n", encoding="utf-8")
    cfg_dir = temp_forge_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"documented_prefixes": ["code_forge"], "excluded_prefixes": [], "excluded_segments": []}),
        encoding="utf-8",
    )
    _git(["add", "."], temp_forge_root)
    _git(["commit", "-m", "init"], temp_forge_root)

    payload = inventory_summary(temp_forge_root, selected_paths={"code_forge/src/code_forge/library"})
    record = next(row for row in payload["records"] if row["path"] == "code_forge/src/code_forge/library")
    assert "code_forge/tests/test_library_db.py" in record["test_references"]
    assert not any("final_docs" in row for row in record["test_references"])

    status = inventory_status(temp_forge_root, selected_paths={"code_forge/src/code_forge/library"})
    assert status["required_directory_count"] >= 1
    assert 0.0 <= status["coverage_ratio"] <= 1.0


def test_route_detection_skips_test_routes(temp_forge_root: Path) -> None:
    _git(["init"], temp_forge_root)
    _git(["config", "user.email", "test@example.com"], temp_forge_root)
    _git(["config", "user.name", "Test"], temp_forge_root)
    target = temp_forge_root / "doc_forge" / "src" / "doc_forge" / "scribe"
    tests = target / "tests"
    target.mkdir(parents=True, exist_ok=True)
    tests.mkdir(parents=True, exist_ok=True)
    (target / "service.py").write_text('from fastapi import FastAPI\napp = FastAPI()\n@app.get("/health")\ndef health():\n    return {"ok": True}\n', encoding="utf-8")
    (tests / "test_service.py").write_text('from fastapi import APIRouter\nrouter = APIRouter()\n@router.get("/coverage")\ndef coverage():\n    return {"ok": True}\n', encoding="utf-8")
    cfg_dir = temp_forge_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"documented_prefixes": ["doc_forge"], "excluded_prefixes": [], "excluded_segments": []}),
        encoding="utf-8",
    )
    _git(["add", "."], temp_forge_root)
    _git(["commit", "-m", "init"], temp_forge_root)

    content = render_directory_readme(temp_forge_root, "doc_forge/src/doc_forge/scribe")
    assert "GET /health" in content
    assert "GET /coverage" not in content


def test_load_policy_merges_override(temp_forge_root: Path) -> None:
    cfg_dir = temp_forge_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"path_overrides": {"lib/eidosian_runtime": {"summary": "Custom summary"}}}),
        encoding="utf-8",
    )
    policy = load_policy(temp_forge_root)
    assert policy["path_overrides"]["lib/eidosian_runtime"]["summary"] == "Custom summary"


def test_inventory_summary_can_scope_to_selected_paths(temp_forge_root: Path) -> None:
    _git(["init"], temp_forge_root)
    _git(["config", "user.email", "test@example.com"], temp_forge_root)
    _git(["config", "user.name", "Test"], temp_forge_root)
    alpha = temp_forge_root / "doc_forge" / "src" / "doc_forge" / "scribe"
    beta = temp_forge_root / "doc_forge" / "src" / "doc_forge" / "utils"
    alpha.mkdir(parents=True, exist_ok=True)
    beta.mkdir(parents=True, exist_ok=True)
    (alpha / "service.py").write_text("x=1\n", encoding="utf-8")
    (beta / "paths.py").write_text("y=2\n", encoding="utf-8")
    cfg_dir = temp_forge_root / "cfg"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / "documentation_policy.json").write_text(
        json.dumps({"documented_prefixes": ["doc_forge"], "excluded_prefixes": [], "excluded_segments": []}),
        encoding="utf-8",
    )
    _git(["add", "."], temp_forge_root)
    _git(["commit", "-m", "init"], temp_forge_root)
    payload = inventory_summary(temp_forge_root, selected_paths={"doc_forge/src/doc_forge/scribe"})
    paths = {row["path"] for row in payload["records"]}
    assert "doc_forge/src/doc_forge/scribe" in paths
    assert "doc_forge/src/doc_forge/utils" not in paths
