from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "scripts" / "run_agencybench_eidos.py"
for extra in (ROOT / "lib", ROOT / "eidos_mcp" / "src", ROOT / "scripts"):
    value = str(extra)
    if value not in sys.path:
        sys.path.insert(0, value)


def _load_module():
    spec = importlib.util.spec_from_file_location("run_agencybench_eidos", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_scenario2_fixture(root: Path) -> Path:
    scenario_root = root / "AgencyBench-v2" / "MCP" / "scenario2"
    desktop = scenario_root / "desktop"
    _write(desktop / "exp_logs" / "2023_aug" / "debug_utils.py")
    _write(desktop / "exp_logs" / "project_alpha" / "raw_data.csv")
    _write(desktop / "exp_logs" / "project_alpha" / "train_model.py")
    _write(desktop / "exp_logs" / "project_alpha" / "README.md")
    _write(desktop / "exp_logs" / "deprecated" / "old_results.csv")
    _write(desktop / "learning" / "2024" / "test_runner.py")
    _write(desktop / "learning" / "2024" / "progress.csv")
    _write(desktop / "learning" / "docs" / "architecture.md")
    _write(desktop / "learning" / "study_notes.md")
    _write(desktop / "music" / "jay_chou" / "favorite_songs.csv")
    _write(desktop / "music" / "music_manifest.md")
    _write(desktop / "old_homebrew" / "config" / "settings.py")
    _write(desktop / "old_homebrew" / "inventory" / "inventory_list.csv")
    _write(desktop / "travel_plan" / "draft_plan.md")
    _write(desktop / "travel_plan" / "budget_calculator.py")
    _write(desktop / "miscellaneous" / "notes.txt")
    (scenario_root / "description.json").write_text(
        json.dumps(
            {
                "workspace_brief": "test fixture",
                "subtask1": "create skeleton",
                "subtask2": "move py",
                "subtask3": "move csv",
                "subtask4": "move md",
                "subtask5": "delete desktop",
            }
        ),
        encoding="utf-8",
    )
    return root


def _complete_workspace(workspace: Path) -> None:
    base = workspace / "desktop_2" / "workspace_v2"
    (base / "dev_bundle" / "tests").mkdir(parents=True, exist_ok=True)
    (base / "dev_bundle" / "source").mkdir(parents=True, exist_ok=True)
    (base / "data_warehouse" / "legacy_archives").mkdir(parents=True, exist_ok=True)
    (base / "data_warehouse" / "active_datasets").mkdir(parents=True, exist_ok=True)
    (base / "knowledge_base").mkdir(parents=True, exist_ok=True)
    moves = {
        workspace
        / "desktop"
        / "exp_logs"
        / "2023_aug"
        / "debug_utils.py": base
        / "dev_bundle"
        / "tests"
        / "debug_utils.py",
        workspace
        / "desktop"
        / "learning"
        / "2024"
        / "test_runner.py": base
        / "dev_bundle"
        / "tests"
        / "test_runner.py",
        workspace
        / "desktop"
        / "exp_logs"
        / "project_alpha"
        / "train_model.py": base
        / "dev_bundle"
        / "source"
        / "train_model.py",
        workspace
        / "desktop"
        / "old_homebrew"
        / "config"
        / "settings.py": base
        / "dev_bundle"
        / "source"
        / "settings.py",
        workspace
        / "desktop"
        / "travel_plan"
        / "budget_calculator.py": base
        / "dev_bundle"
        / "source"
        / "budget_calculator.py",
        workspace
        / "desktop"
        / "exp_logs"
        / "project_alpha"
        / "raw_data.csv": base
        / "data_warehouse"
        / "legacy_archives"
        / "raw_data.csv",
        workspace
        / "desktop"
        / "exp_logs"
        / "deprecated"
        / "old_results.csv": base
        / "data_warehouse"
        / "legacy_archives"
        / "old_results.csv",
        workspace
        / "desktop"
        / "old_homebrew"
        / "inventory"
        / "inventory_list.csv": base
        / "data_warehouse"
        / "legacy_archives"
        / "inventory_list.csv",
        workspace
        / "desktop"
        / "learning"
        / "2024"
        / "progress.csv": base
        / "data_warehouse"
        / "active_datasets"
        / "progress.csv",
        workspace
        / "desktop"
        / "music"
        / "jay_chou"
        / "favorite_songs.csv": base
        / "data_warehouse"
        / "active_datasets"
        / "favorite_songs.csv",
        workspace
        / "desktop"
        / "exp_logs"
        / "project_alpha"
        / "README.md": base
        / "knowledge_base"
        / "project_alpha_README.md",
        workspace
        / "desktop"
        / "learning"
        / "docs"
        / "architecture.md": base
        / "knowledge_base"
        / "docs_architecture.md",
        workspace / "desktop" / "learning" / "study_notes.md": base / "knowledge_base" / "learning_study_notes.md",
        workspace / "desktop" / "music" / "music_manifest.md": base / "knowledge_base" / "music_music_manifest.md",
        workspace / "desktop" / "travel_plan" / "draft_plan.md": base / "knowledge_base" / "travel_plan_draft_plan.md",
    }
    for src, dst in moves.items():
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    if (workspace / "desktop").exists():
        shutil.rmtree(workspace / "desktop")


def test_verify_scenario2_contracts(tmp_path: Path) -> None:
    module = _load_module()
    fixture_root = _build_scenario2_fixture(tmp_path / "agencybench")
    workspace = module.prepare_scenario2_workspace(
        fixture_root / "AgencyBench-v2" / "MCP" / "scenario2",
        tmp_path / "run",
    )
    ok, _ = module.verify_scenario2_subtask1(workspace)
    assert ok is False
    _complete_workspace(workspace)
    for verifier in [
        module.verify_scenario2_subtask1,
        module.verify_scenario2_subtask2,
        module.verify_scenario2_subtask3,
        module.verify_scenario2_subtask4,
        module.verify_scenario2_subtask5,
    ]:
        ok, msg = verifier(workspace)
        assert ok, msg


def test_run_scenario2_writes_external_benchmark_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    fixture_root = _build_scenario2_fixture(tmp_path / "agencybench")
    monkeypatch.setattr(module, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(module, "REPORT_ROOT", tmp_path / "reports")
    result = module.run_scenario2(
        agencybench_root=fixture_root,
        repo_root=ROOT,
        model="qwen3.5:2b",
        attempts_per_step=1,
        timeout_sec=30.0,
        keep_alive="1h",
        engine="deterministic",
    )
    assert result["normalized"]["score"] == 1.0
    latest = tmp_path / "reports" / f"{module.SCENARIO2.suite_name}_deterministic" / "latest.json"
    assert latest.exists()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["execution_mode"] == "local_run"
    assert payload["metrics"]["tasks_passed"] == 5


def test_verify_scenario1_contracts(monkeypatch) -> None:
    module = _load_module()
    target = module.GitHubBenchmarkTarget(
        owner="Ace1928",
        repo="eidos-agencybench-s1-test",
        repo_url="https://github.com/Ace1928/eidos-agencybench-s1-test",
        worktree=ROOT,
    )

    monkeypatch.setattr(
        module,
        "_gh_issue_view",
        lambda _full_name, _issue_number: {
            "number": 12,
            "title": module.SCENARIO1_ISSUE_TITLE,
            "body": module.SCENARIO1_ISSUE_BODY,
            "labels": [{"name": "triage"}, {"name": "meta"}],
        },
    )
    monkeypatch.setattr(module, "_gh_branch_exists", lambda _full_name, branch: branch == "config/issue-templates")
    monkeypatch.setattr(
        module,
        "_gh_file_content",
        lambda _full_name, _path, ref: module.SCENARIO1_TEMPLATE_BODY if ref == "config/issue-templates" else "",
    )
    monkeypatch.setattr(
        module,
        "_gh_issue_comments",
        lambda _full_name, _issue_number: [
            {"body": module.SCENARIO1_COMMENT_TEMPLATE.format(template=module.SCENARIO1_TEMPLATE_BODY.rstrip())}
        ],
    )
    monkeypatch.setattr(
        module,
        "_gh_pr_view",
        lambda _full_name, pr_number: {
            "number": pr_number,
            "title": module.SCENARIO1_PR_TITLE,
            "body": module.SCENARIO1_PR_BODY_TEMPLATE.format(issue_number=12),
            "labels": [{"name": "configuration"}, {"name": "dependencies"}],
            "headRefName": "config/issue-templates",
            "baseRefName": "main",
        },
    )

    ok, msg = module.verify_scenario1_subtask1(target, 12)
    assert ok, msg
    ok, msg = module.verify_scenario1_subtask2(target)
    assert ok, msg
    ok, msg = module.verify_scenario1_subtask3(target)
    assert ok, msg

    monkeypatch.setattr(
        module,
        "_gh_issue_view",
        lambda _full_name, _issue_number: {
            "number": 12,
            "title": module.SCENARIO1_ISSUE_TITLE,
            "body": module.SCENARIO1_ISSUE_BODY,
            "labels": [{"name": "meta"}, {"name": "in-progress"}],
        },
    )
    ok, msg = module.verify_scenario1_subtask4(target, 12)
    assert ok, msg
    ok, msg = module.verify_scenario1_subtask5(target, 12, 34)
    assert ok, msg


def test_run_scenario1_writes_external_benchmark_artifacts(tmp_path: Path, monkeypatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "RUNTIME_ROOT", tmp_path / "runtime")
    monkeypatch.setattr(module, "REPORT_ROOT", tmp_path / "reports")
    monkeypatch.setattr(
        module,
        "_execute_scenario1_deterministic",
        lambda **_kwargs: {
            "target": {
                "owner": "Ace1928",
                "repo": "eidos-agencybench-s1-test",
                "repo_url": "https://github.com/Ace1928/eidos-agencybench-s1-test",
                "visibility": "private",
                "worktree": str(tmp_path / "repo"),
            },
            "issue_number": 12,
            "pr_number": 34,
            "verification": [{"step": f"subtask{i}", "success": True, "message": "ok"} for i in range(1, 6)],
        },
    )
    result = module.run_scenario1(
        repo_root=ROOT,
        model="qwen3.5:2b",
        engine="deterministic",
        repo_visibility="private",
    )
    assert result["normalized"]["score"] == 1.0
    latest = tmp_path / "reports" / "agencybench_eidos_scenario1_deterministic" / "latest.json"
    assert latest.exists()
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["metrics"]["tasks_passed"] == 5
    assert payload["status"] == "green"
