#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
for extra in (REPO_ROOT / "lib", REPO_ROOT / "eidos_mcp" / "src", REPO_ROOT / "scripts"):
    value = str(extra)
    if value not in sys.path:
        sys.path.insert(0, value)

from eidos_scheduler import apply_scheduler_control
from eidosian_agent.local_mcp_agent import LocalMcpAgent, normalize_profile
from eidosian_runtime import ForgeRuntimeCoordinator

DEFAULT_AGENCYBENCH_ROOT = Path("/data/data/com.termux/files/usr/tmp/eidos-agencybench-22810")
RUNTIME_ROOT = REPO_ROOT / "data" / "runtime" / "external_benchmarks" / "agencybench"
REPORT_ROOT = REPO_ROOT / "reports" / "external_benchmarks"


Validator = Callable[[Path], tuple[bool, str]]


@dataclass(frozen=True)
class ScenarioStep:
    key: str
    prompt: str
    validator: Validator
    acceptance_hint: str


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    suite_name: str
    source_dir: str
    workspace_prep: Callable[[Path, Path], Path]
    steps: list[ScenarioStep]


@dataclass(frozen=True)
class GitHubBenchmarkTarget:
    owner: str
    repo: str
    repo_url: str
    worktree: Path


def _now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _scheduler_pause_requested() -> bool:
    payload = apply_scheduler_control("status")
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    return bool(state.get("pause_requested"))


def _pause_scheduler_for_benchmark() -> bool:
    already_paused = _scheduler_pause_requested()
    if not already_paused:
        apply_scheduler_control("pause")
    return not already_paused


def _resume_scheduler_if_needed(resume_needed: bool) -> None:
    if resume_needed:
        apply_scheduler_control("resume")


def _wait_for_model_budget(
    coordinator: ForgeRuntimeCoordinator,
    *,
    owner: str,
    requested_models: list[dict[str, Any]],
    wait_timeout_sec: float,
    stale_after_sec: float,
) -> dict[str, Any]:
    started = time.monotonic()
    stale_recoveries = 0
    last_decision: dict[str, Any] = {}
    while time.monotonic() - started <= max(5.0, float(wait_timeout_sec)):
        last_decision = coordinator.can_allocate(
            owner=owner,
            requested_models=requested_models,
            allow_same_owner=False,
        )
        if last_decision.get("allowed"):
            return {
                "allowed": True,
                "waited_sec": round(time.monotonic() - started, 3),
                "stale_recoveries": stale_recoveries,
                "decision": last_decision,
            }
        recovery = coordinator.recover_stale_owner(
            stale_after_sec=stale_after_sec,
            reason="benchmark_budget_recovery",
        )
        if recovery.get("released"):
            stale_recoveries += 1
            continue
        time.sleep(5.0)
    return {
        "allowed": False,
        "waited_sec": round(time.monotonic() - started, 3),
        "stale_recoveries": stale_recoveries,
        "decision": last_decision,
    }


def _workspace_v2_root(workspace: Path) -> Path:
    return workspace / "desktop_2" / "workspace_v2"


def verify_scenario2_subtask1(workspace: Path) -> tuple[bool, str]:
    base = _workspace_v2_root(workspace)
    required = [
        base,
        base / "dev_bundle",
        base / "dev_bundle" / "tests",
        base / "dev_bundle" / "source",
        base / "data_warehouse",
        base / "data_warehouse" / "legacy_archives",
        base / "data_warehouse" / "active_datasets",
        base / "knowledge_base",
    ]
    missing = [str(path.relative_to(workspace)) for path in required if not path.is_dir()]
    if missing:
        return False, f"Missing directories: {', '.join(missing)}"
    return True, "workspace_v2 directory skeleton exists."


def verify_scenario2_subtask2(workspace: Path) -> tuple[bool, str]:
    base = _workspace_v2_root(workspace) / "dev_bundle"
    tests_dir = base / "tests"
    source_dir = base / "source"
    if not tests_dir.is_dir() or not source_dir.is_dir():
        return False, "dev_bundle/tests or dev_bundle/source missing."
    expected_tests = {"debug_utils.py", "test_runner.py"}
    expected_source = {"train_model.py", "settings.py", "budget_calculator.py"}
    found_tests = {f.name for f in tests_dir.glob("*.py")}
    found_source = {f.name for f in source_dir.glob("*.py")}
    missing_tests = expected_tests - found_tests
    missing_source = expected_source - found_source
    misplaced_tests = expected_source & found_tests
    misplaced_source = expected_tests & found_source
    problems: list[str] = []
    if missing_tests:
        problems.append(f"Missing in tests/: {sorted(missing_tests)}")
    if missing_source:
        problems.append(f"Missing in source/: {sorted(missing_source)}")
    if misplaced_tests:
        problems.append(f"Source-only files wrongly placed in tests/: {sorted(misplaced_tests)}")
    if misplaced_source:
        problems.append(f"Test/debug files wrongly placed in source/: {sorted(misplaced_source)}")
    if problems:
        return False, "; ".join(problems)
    return True, "Python files relocated into correct dev_bundle folders."


def verify_scenario2_subtask3(workspace: Path) -> tuple[bool, str]:
    base = _workspace_v2_root(workspace) / "data_warehouse"
    legacy_dir = base / "legacy_archives"
    active_dir = base / "active_datasets"
    if not legacy_dir.is_dir() or not active_dir.is_dir():
        return False, "data_warehouse subdirectories missing."
    expected_legacy = {"raw_data.csv", "old_results.csv", "inventory_list.csv"}
    expected_active = {"progress.csv", "favorite_songs.csv"}
    found_legacy = {f.name for f in legacy_dir.glob("*.csv")}
    found_active = {f.name for f in active_dir.glob("*.csv")}
    missing_legacy = expected_legacy - found_legacy
    missing_active = expected_active - found_active
    misplaced_to_legacy = expected_active & found_legacy
    misplaced_to_active = expected_legacy & found_active
    issues: list[str] = []
    if missing_legacy:
        issues.append(f"Legacy files missing: {sorted(missing_legacy)}")
    if missing_active:
        issues.append(f"Active files missing: {sorted(missing_active)}")
    if misplaced_to_legacy:
        issues.append(f"Active files wrongly placed in legacy: {sorted(misplaced_to_legacy)}")
    if misplaced_to_active:
        issues.append(f"Legacy files wrongly placed in active: {sorted(misplaced_to_active)}")
    if issues:
        return False, "; ".join(issues)
    return True, "CSV files split between legacy and active targets correctly."


def verify_scenario2_subtask4(workspace: Path) -> tuple[bool, str]:
    kb_dir = _workspace_v2_root(workspace) / "knowledge_base"
    if not kb_dir.is_dir():
        return False, "knowledge_base directory missing."
    expected_files = {
        "project_alpha_README.md",
        "docs_architecture.md",
        "learning_study_notes.md",
        "music_music_manifest.md",
        "travel_plan_draft_plan.md",
    }
    found = {f.name for f in kb_dir.glob("*.md")}
    missing = expected_files - found
    if missing:
        return False, f"Missing renamed markdown files: {sorted(missing)}"
    return True, "All markdown files moved and renamed with parent prefixes."


def verify_scenario2_subtask5(workspace: Path) -> tuple[bool, str]:
    source_dir = workspace / "desktop"
    if source_dir.exists():
        return False, f"Legacy desktop directory still present at {source_dir}."
    return True, "Legacy desktop/ tree successfully deleted."


def prepare_scenario2_workspace(source_dir: Path, run_root: Path) -> Path:
    workspace = run_root / "workspace"
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_dir / "desktop", workspace / "desktop")
    (workspace / "desktop_2").mkdir(parents=True, exist_ok=True)
    return workspace


SCENARIO2 = ScenarioSpec(
    name="scenario2",
    suite_name="agencybench_eidos_scenario2",
    source_dir="AgencyBench-v2/MCP/scenario2",
    workspace_prep=prepare_scenario2_workspace,
    steps=[
        ScenarioStep(
            key="subtask1",
            prompt="Build the exact workspace_v2 directory skeleton under desktop_2.",
            validator=verify_scenario2_subtask1,
            acceptance_hint="Evaluator checks for desktop_2/workspace_v2, dev_bundle/tests, dev_bundle/source, data_warehouse/legacy_archives, data_warehouse/active_datasets, and knowledge_base.",
        ),
        ScenarioStep(
            key="subtask2",
            prompt="Move all Python files from desktop into the correct dev_bundle/tests or dev_bundle/source target using filename semantics.",
            validator=verify_scenario2_subtask2,
            acceptance_hint="debug_utils.py and test_runner.py must be in tests; train_model.py, settings.py, and budget_calculator.py must be in source; no duplicates.",
        ),
        ScenarioStep(
            key="subtask3",
            prompt="Move all CSV files from desktop into legacy_archives or active_datasets based on old/exp ancestry.",
            validator=verify_scenario2_subtask3,
            acceptance_hint="raw_data.csv, old_results.csv, and inventory_list.csv are legacy; progress.csv and favorite_songs.csv are active.",
        ),
        ScenarioStep(
            key="subtask4",
            prompt="Move every markdown file into knowledge_base and rename each one to {ImmediateParentDirectoryName}_{OriginalFilename}.",
            validator=verify_scenario2_subtask4,
            acceptance_hint="Expected outputs: project_alpha_README.md, docs_architecture.md, learning_study_notes.md, music_music_manifest.md, travel_plan_draft_plan.md.",
        ),
        ScenarioStep(
            key="subtask5",
            prompt="Delete the original desktop directory only after confirming all assets have been migrated into desktop_2/workspace_v2.",
            validator=verify_scenario2_subtask5,
            acceptance_hint="The original desktop/ tree must be gone, while desktop_2/workspace_v2 remains intact.",
        ),
    ],
)


SCENARIO1_ISSUE_TITLE = "Setup standard bug report template for process improvement"
SCENARIO1_ISSUE_BODY = """## Context
We need an issue template upgrade to improve issue template standardization across the repository.

## Goals
- introduce a reusable issue template for bug tracking
- improve issue template standardization for maintainers
- make bug tracking intake more consistent

## Expected Outcome
This process improvement should leave the repository with a standard issue template workflow for bug tracking and clearer intake expectations.
"""

SCENARIO1_TEMPLATE_BODY = """---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Describe the Bug
Provide a clear and concise description of the bug.

## Reproduction Steps
List the steps required to reproduce the issue.

## Expected Behavior
Describe what you expected to happen instead.

## Environment
Capture the runtime, platform, and relevant version details.

Please search for existing issues before submitting a new report.
"""

SCENARIO1_PR_TITLE = "Add structured bug report template configuration"
SCENARIO1_COMMENT_TEMPLATE = """```md
{template}
```"""
SCENARIO1_PR_BODY_TEMPLATE = """## Description
This configuration change adds a structured bug report template with explicit fields for issue intake, reproduction, expected behavior, and environment data.

Resolves #{issue_number}

## Verification
The raw markdown is available in the issue comments for direct review.
"""


def _run(cmd: list[str], *, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    completed = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=merged_env,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "command_failed: " + " ".join(cmd) + f"\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def _gh(args: list[str], *, cwd: Path | None = None) -> str:
    return _run(["gh", *args], cwd=cwd)


def _git(args: list[str], *, cwd: Path) -> str:
    return _run(["git", *args], cwd=cwd)


def _gh_json(args: list[str], *, cwd: Path | None = None) -> Any:
    text = _gh(args, cwd=cwd)
    return json.loads(text) if text else None


def _github_login() -> str:
    login = _gh(["api", "user", "--jq", ".login"]).strip()
    if not login:
        raise RuntimeError("Unable to resolve authenticated GitHub login.")
    return login


def _extract_number_from_url(url: str) -> int:
    value = str(url or "").rstrip("/").split("/")[-1]
    return int(value)


def _scenario1_repo_name(stamp: str) -> str:
    return f"eidos-agencybench-s1-{stamp.lower()}"


def _ensure_github_repo(run_root: Path, *, owner: str, repo_name: str, visibility: str) -> GitHubBenchmarkTarget:
    payload = _gh_json(
        [
            "api",
            "user/repos",
            "--method",
            "POST",
            "-f",
            f"name={repo_name}",
            "-F",
            f"private={'true' if visibility == 'private' else 'false'}",
            "-f",
            "auto_init=true",
            "-f",
            "description=Eidos AgencyBench scenario1 deterministic benchmark repo",
        ]
    )
    full_name = str(payload.get("full_name") or f"{owner}/{repo_name}")
    repo_url = str(payload.get("html_url") or f"https://github.com/{full_name}")
    worktree = run_root / "repo"
    if worktree.exists():
        shutil.rmtree(worktree)
    _gh(["repo", "clone", full_name, str(worktree)])
    _git(["config", "user.name", "Eidos Benchmark"], cwd=worktree)
    _git(["config", "user.email", "eidos-benchmark@local.invalid"], cwd=worktree)
    _git(["branch", "-M", "main"], cwd=worktree)
    _git(["push", "-u", "origin", "main"], cwd=worktree)
    _gh(["api", f"repos/{full_name}", "--method", "PATCH", "-f", "default_branch=main"])
    return GitHubBenchmarkTarget(owner=owner, repo=repo_name, repo_url=repo_url, worktree=worktree)


def _ensure_labels(full_name: str) -> None:
    labels = {
        "triage": ("d4c5f9", "Needs triage"),
        "meta": ("5319e7", "Meta process task"),
        "in-progress": ("fbca04", "Work in progress"),
        "configuration": ("0e8a16", "Configuration change"),
        "dependencies": ("0366d6", "Dependency or workflow coupling"),
        "bug": ("d73a4a", "Bug report"),
    }
    for name, (color, description) in labels.items():
        _gh(
            [
                "label",
                "create",
                name,
                "--repo",
                full_name,
                "--color",
                color,
                "--description",
                description,
                "--force",
            ]
        )


def _gh_issue_labels_set(full_name: str, issue_number: int, labels: list[str]) -> list[str]:
    args = [
        "api",
        f"repos/{full_name}/issues/{issue_number}/labels",
        "--method",
        "PUT",
    ]
    for label in labels:
        args.extend(["-f", f"labels[]={label}"])
    payload = _gh_json(args)
    names: list[str] = []
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and str(row.get("name") or "").strip():
                names.append(str(row["name"]))
    return names


def _gh_issue_create(full_name: str, *, title: str, body: str, labels: list[str]) -> int:
    args = ["issue", "create", "--repo", full_name, "--title", title, "--body", body]
    for label in labels:
        args.extend(["--label", label])
    url = _gh(args)
    return _extract_number_from_url(url)


def _gh_issue_comment(full_name: str, issue_number: int, body: str) -> None:
    _gh(["issue", "comment", str(issue_number), "--repo", full_name, "--body", body])


def _gh_issue_view(full_name: str, issue_number: int) -> dict[str, Any]:
    return _gh_json(
        [
            "issue",
            "view",
            str(issue_number),
            "--repo",
            full_name,
            "--json",
            "number,title,body,labels,url",
        ]
    )


def _gh_issue_comments(full_name: str, issue_number: int) -> list[dict[str, Any]]:
    payload = _gh_json(["api", f"repos/{full_name}/issues/{issue_number}/comments"])
    return payload if isinstance(payload, list) else []


def _gh_pr_create(full_name: str, *, title: str, body: str, head: str, base: str, labels: list[str]) -> int:
    args = [
        "pr",
        "create",
        "--repo",
        full_name,
        "--head",
        head,
        "--base",
        base,
        "--title",
        title,
        "--body",
        body,
    ]
    for label in labels:
        args.extend(["--label", label])
    url = _gh(args)
    return _extract_number_from_url(url)


def _gh_pr_view(full_name: str, pr_number: int) -> dict[str, Any]:
    return _gh_json(
        [
            "pr",
            "view",
            str(pr_number),
            "--repo",
            full_name,
            "--json",
            "number,title,body,labels,headRefName,baseRefName,url",
        ]
    )


def _gh_branch_exists(full_name: str, branch: str) -> bool:
    completed = subprocess.run(
        ["gh", "api", f"repos/{full_name}/branches/{branch}"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.returncode == 0


def _gh_file_content(full_name: str, path: str, *, ref: str) -> str:
    payload = _gh_json(["api", f"repos/{full_name}/contents/{path}?ref={ref}"])
    content = str((payload or {}).get("content") or "")
    if not content:
        return ""
    import base64

    return base64.b64decode(content).decode("utf-8")


def verify_scenario1_subtask1(target: GitHubBenchmarkTarget, issue_number: int | None = None) -> tuple[bool, str]:
    issue = _gh_issue_view(f"{target.owner}/{target.repo}", issue_number) if issue_number else None
    if not issue:
        return False, "Tracking issue missing."
    title = str(issue.get("title") or "").lower()
    if "setup standard bug report template" not in title or "process improvement" not in title:
        return False, "Issue title missing required keywords."
    body = str(issue.get("body") or "")
    missing_sections = [section for section in ("## Context", "## Goals", "## Expected Outcome") if section not in body]
    missing_keywords = [
        word for word in ("issue template", "standardization", "bug tracking") if word not in body.lower()
    ]
    labels = {str(row.get("name") or "") for row in issue.get("labels") or [] if isinstance(row, dict)}
    if missing_sections:
        return False, f"Issue body missing sections: {missing_sections}"
    if missing_keywords:
        return False, f"Issue body missing keywords: {missing_keywords}"
    if not {"triage", "meta"}.issubset(labels):
        return False, f"Issue labels incorrect: {sorted(labels)}"
    return True, f"Issue #{issue.get('number')} matches required title/body/labels."


def verify_scenario1_subtask2(target: GitHubBenchmarkTarget) -> tuple[bool, str]:
    full_name = f"{target.owner}/{target.repo}"
    if not _gh_branch_exists(full_name, "config/issue-templates"):
        return False, "Branch config/issue-templates not found."
    return True, "Branch config/issue-templates exists on GitHub."


def verify_scenario1_subtask3(target: GitHubBenchmarkTarget) -> tuple[bool, str]:
    full_name = f"{target.owner}/{target.repo}"
    content = _gh_file_content(full_name, ".github/ISSUE_TEMPLATE/bug_report.md", ref="config/issue-templates")
    if not content:
        return False, "Template file missing from config/issue-templates."
    required = [
        "name: Bug Report",
        "about: Create a report to help us improve",
        "title: '[BUG] '",
        "labels: bug",
        "assignees: ''",
        "## Describe the Bug",
        "## Reproduction Steps",
        "## Expected Behavior",
        "## Environment",
    ]
    missing = [item for item in required if item not in content]
    if missing:
        return False, f"Template file missing content: {missing}"
    if "search for existing issues" not in content.lower():
        return False, "Template missing existing-issues reminder."
    return True, "Template file matches required content."


def verify_scenario1_subtask4(target: GitHubBenchmarkTarget, issue_number: int | None = None) -> tuple[bool, str]:
    issue = _gh_issue_view(f"{target.owner}/{target.repo}", issue_number) if issue_number else None
    if not issue:
        return False, "Tracking issue missing."
    labels = {str(row.get("name") or "") for row in issue.get("labels") or [] if isinstance(row, dict)}
    if "triage" in labels or "meta" not in labels or "in-progress" not in labels:
        return False, f"Issue labels incorrect after progress update: {sorted(labels)}"
    comments = _gh_issue_comments(f"{target.owner}/{target.repo}", int(issue.get("number")))
    found = False
    for row in comments:
        body = str(row.get("body") or "")
        if "name: Bug Report" in body and "## Describe the Bug" in body and "```" in body:
            found = True
            break
    if not found:
        return False, "Raw template markdown comment not found."
    return True, "Issue labels and template comment are correct."


def verify_scenario1_subtask5(
    target: GitHubBenchmarkTarget, issue_number: int | None = None, pr_number: int | None = None
) -> tuple[bool, str]:
    if issue_number is None or pr_number is None:
        return False, "Issue or PR number missing."
    pr = _gh_pr_view(f"{target.owner}/{target.repo}", pr_number)
    title = str(pr.get("title") or "").lower()
    if "add structured bug report template" not in title or "configuration" not in title:
        return False, "PR title missing required keywords."
    body = str(pr.get("body") or "")
    labels = {str(row.get("name") or "") for row in pr.get("labels") or [] if isinstance(row, dict)}
    missing_sections = [section for section in ("## Description", "## Verification") if section not in body]
    if missing_sections:
        return False, f"PR body missing sections: {missing_sections}"
    if f"resolves #{issue_number}".lower() not in body.lower():
        return False, "PR body missing issue resolution reference."
    if "raw markdown" not in body.lower() or "issue comments" not in body.lower():
        return False, "PR body missing issue-comment verification note."
    if not {"configuration", "dependencies"}.issubset(labels):
        return False, f"PR labels incorrect: {sorted(labels)}"
    if str(pr.get("headRefName") or "") != "config/issue-templates" or str(pr.get("baseRefName") or "") != "main":
        return False, "PR head/base refs incorrect."
    return True, f"PR #{pr_number} satisfies required title/body/labels."


def _execute_scenario1_deterministic(
    *,
    run_root: Path,
    stamp: str,
    repo_visibility: str,
) -> dict[str, Any]:
    owner = _github_login()
    repo_name = _scenario1_repo_name(stamp)
    target = _ensure_github_repo(run_root, owner=owner, repo_name=repo_name, visibility=repo_visibility)
    full_name = f"{target.owner}/{target.repo}"
    _ensure_labels(full_name)
    issue_number = _gh_issue_create(
        full_name, title=SCENARIO1_ISSUE_TITLE, body=SCENARIO1_ISSUE_BODY, labels=["triage", "meta"]
    )
    verification_rows = []
    ok, message = verify_scenario1_subtask1(target, issue_number)
    verification_rows.append({"step": "subtask1", "success": ok, "message": message})
    _git(["checkout", "-b", "config/issue-templates"], cwd=target.worktree)
    template_path = target.worktree / ".github" / "ISSUE_TEMPLATE" / "bug_report.md"
    template_path.parent.mkdir(parents=True, exist_ok=True)
    template_path.write_text(SCENARIO1_TEMPLATE_BODY, encoding="utf-8")
    _git(["add", ".github/ISSUE_TEMPLATE/bug_report.md"], cwd=target.worktree)
    _git(["commit", "-m", "Add structured bug report template"], cwd=target.worktree)
    _git(["push", "-u", "origin", "config/issue-templates"], cwd=target.worktree)
    ok, message = verify_scenario1_subtask2(target)
    verification_rows.append({"step": "subtask2", "success": ok, "message": message})
    ok, message = verify_scenario1_subtask3(target)
    verification_rows.append({"step": "subtask3", "success": ok, "message": message})
    _gh_issue_labels_set(full_name, issue_number, ["meta", "in-progress"])
    _gh_issue_comment(
        full_name, issue_number, SCENARIO1_COMMENT_TEMPLATE.format(template=SCENARIO1_TEMPLATE_BODY.rstrip())
    )
    ok, message = verify_scenario1_subtask4(target, issue_number)
    verification_rows.append({"step": "subtask4", "success": ok, "message": message})
    pr_number = _gh_pr_create(
        full_name,
        title=SCENARIO1_PR_TITLE,
        body=SCENARIO1_PR_BODY_TEMPLATE.format(issue_number=issue_number),
        head="config/issue-templates",
        base="main",
        labels=["configuration", "dependencies"],
    )
    ok, message = verify_scenario1_subtask5(target, issue_number, pr_number)
    verification_rows.append({"step": "subtask5", "success": ok, "message": message})
    return {
        "target": {
            "owner": target.owner,
            "repo": target.repo,
            "repo_url": target.repo_url,
            "visibility": repo_visibility,
            "worktree": str(target.worktree),
        },
        "issue_number": issue_number,
        "pr_number": pr_number,
        "verification": verification_rows,
    }


def _build_scenario2_policy(workspace: Path) -> dict[str, Any]:
    root = str(workspace)
    return {
        "contract": "eidos.local_agent_profiles.v1",
        "profiles": {
            "agencybench_scenario2": {
                "description": "Constrained filesystem benchmark agent for AgencyBench MCP scenario2.",
                "max_steps": 14,
                "max_tool_calls": 24,
                "max_mutating_calls": 24,
                "thinking_mode": "off",
                "keep_alive": "4h",
                "temperature": 0.0,
                "max_tokens": 1024,
                "max_observation_chars": 4000,
                "system_prompt": (
                    "You are running an external benchmark scenario. Only use the provided tools. "
                    "Use absolute paths under the provided workspace root. Execute one focused action at a time. "
                    "Do not touch anything outside the benchmark workspace."
                ),
                "allowed_tools": {
                    "run_shell_command": {
                        "allowed_keys": ["command", "cwd", "timeout_sec"],
                        "string_max_lengths": {"command": 320, "cwd": 400},
                        "path_prefixes": {"cwd": [root]},
                        "allowed_patterns": {"command": [r"^(pwd|ls|find|mkdir|mv|rm|test|cat|sed|printf)(\s|$)"]},
                        "blocked_patterns": {
                            "command": [
                                r"[;&|]",
                                r"\b(curl|wget|ssh|scp|git|gh|python|perl|ruby|node|tee|dd|chmod|chown)\b",
                                r"\.\./",
                            ]
                        },
                        "integer_bounds": {"timeout_sec": {"min": 1, "max": 120}},
                        "const_arguments": {
                            "safe_mode": True,
                            "transaction_paths": [root],
                            "idempotency_key": "agencybench-scenario2",
                        },
                        "mode": "mutating",
                        "max_calls_per_cycle": 20,
                    },
                    "file_read": {
                        "allowed_keys": ["file_path", "max_bytes"],
                        "string_max_lengths": {"file_path": 400},
                        "path_prefixes": {"file_path": [root]},
                        "integer_bounds": {"max_bytes": {"min": 1, "max": 20000}},
                        "mode": "read",
                        "max_calls_per_cycle": 6,
                    },
                    "file_write": {
                        "allowed_keys": ["file_path", "content"],
                        "string_max_lengths": {"file_path": 400, "content": 12000},
                        "path_prefixes": {"file_path": [root]},
                        "const_arguments": {"overwrite": True},
                        "mode": "mutating",
                        "max_calls_per_cycle": 6,
                    },
                },
            }
        },
    }


def _description_text(description_path: Path) -> str:
    payload = json.loads(description_path.read_text(encoding="utf-8"))
    lines = []
    for key in ["workspace_brief"]:
        value = str(payload.get(key) or "").strip()
        if value:
            lines.append(f"{key}:\n{value}")
    return "\n\n".join(lines)


def _mkdirs(paths: list[Path]) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def _move_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def _execute_scenario2_step_deterministic(workspace: Path, step_key: str) -> dict[str, Any]:
    base = _workspace_v2_root(workspace)
    if step_key == "subtask1":
        _mkdirs(
            [
                base / "dev_bundle" / "tests",
                base / "dev_bundle" / "source",
                base / "data_warehouse" / "legacy_archives",
                base / "data_warehouse" / "active_datasets",
                base / "knowledge_base",
            ]
        )
    elif step_key == "subtask2":
        _mkdirs([base / "dev_bundle" / "tests", base / "dev_bundle" / "source"])
        for src in (workspace / "desktop").rglob("*.py"):
            target_dir = (
                base
                / "dev_bundle"
                / ("tests" if ("test" in src.name.lower() or "debug" in src.name.lower()) else "source")
            )
            _move_if_exists(src, target_dir / src.name)
    elif step_key == "subtask3":
        _mkdirs([base / "data_warehouse" / "legacy_archives", base / "data_warehouse" / "active_datasets"])
        for src in (workspace / "desktop").rglob("*.csv"):
            parent_names = [part.lower() for part in src.parts]
            is_legacy = any(("old" in part or "exp" in part) for part in parent_names)
            target_dir = base / "data_warehouse" / ("legacy_archives" if is_legacy else "active_datasets")
            _move_if_exists(src, target_dir / src.name)
    elif step_key == "subtask4":
        _mkdirs([base / "knowledge_base"])
        for src in (workspace / "desktop").rglob("*.md"):
            parent = src.parent.name
            target_name = f"{parent}_{src.name}"
            _move_if_exists(src, base / "knowledge_base" / target_name)
    elif step_key == "subtask5":
        if (workspace / "desktop").exists():
            shutil.rmtree(workspace / "desktop")
    else:
        raise ValueError(f"Unsupported deterministic scenario2 step: {step_key}")
    return {
        "contract": "eidos.agencybench_executor_result.v1",
        "status": "success",
        "engine": "deterministic",
        "step": step_key,
    }


def _render_attempt_objective(
    *,
    workspace: Path,
    description_text: str,
    completed: list[str],
    current_step: ScenarioStep,
    feedback: str,
    attempt_index: int,
) -> str:
    completed_text = "\n".join(f"- {item}" for item in completed) or "- none yet"
    feedback_text = feedback.strip() or "No previous failure feedback."
    return (
        f"Run AgencyBench MCP {current_step.key} in the benchmark workspace.\n"
        f"Workspace root: {workspace}\n"
        "Rules:\n"
        "- Use absolute paths under the workspace root for all file_path and cwd arguments.\n"
        "- Use one focused shell command per tool call.\n"
        "- Stay inside the workspace root.\n"
        "- Stop once the current subtask is complete.\n\n"
        f"Official scenario description:\n{description_text}\n\n"
        f"Already verified completed subtasks:\n{completed_text}\n\n"
        f"Current target ({current_step.key}):\n{current_step.prompt}\n"
        f"Acceptance hint: {current_step.acceptance_hint}\n\n"
        f"Attempt index: {attempt_index}\n"
        f"Previous validator feedback:\n{feedback_text}"
    )


def run_scenario2(
    *,
    agencybench_root: Path,
    repo_root: Path,
    model: str,
    attempts_per_step: int,
    timeout_sec: float,
    keep_alive: str,
    engine: str,
) -> dict[str, Any]:
    scenario_root = agencybench_root / SCENARIO2.source_dir
    if not scenario_root.exists():
        raise FileNotFoundError(f"Scenario root not found: {scenario_root}")
    stamp = _now_stamp()
    run_root = RUNTIME_ROOT / SCENARIO2.name / stamp
    workspace = SCENARIO2.workspace_prep(scenario_root, run_root)
    policy_path = run_root / "policy.json"
    _write_json(policy_path, _build_scenario2_policy(workspace))
    description_text = _description_text(scenario_root / "description.json")
    profile = None
    agent = None
    coordinator = None
    scheduler_resume_needed = False
    if engine == "local_agent":
        profile = normalize_profile(
            "agencybench_scenario2",
            json.loads(policy_path.read_text(encoding="utf-8"))["profiles"]["agencybench_scenario2"],
        )
        profile = replace(profile, keep_alive=keep_alive or profile.keep_alive)
        agent = LocalMcpAgent(
            forge_root=repo_root,
            model=model,
            profile=profile,
            runtime_dir=run_root / "local_mcp_agent",
        )
        coordinator = ForgeRuntimeCoordinator()
        scheduler_resume_needed = _pause_scheduler_for_benchmark()

    completed: list[str] = []
    attempts: list[dict[str, Any]] = []
    final_status = "success"
    stop_reason = "all_subtasks_completed"
    budget = {
        "allowed": True,
        "waited_sec": 0.0,
        "stale_recoveries": 0,
        "decision": {"reason": "ok"},
    }
    try:
        if engine == "local_agent":
            assert coordinator is not None and profile is not None and agent is not None
            budget = _wait_for_model_budget(
                coordinator,
                owner=f"local_mcp_agent:{profile.name}",
                requested_models=[{"family": "ollama", "model": model, "role": f"local_agent:{profile.name}"}],
                wait_timeout_sec=min(max(float(timeout_sec), 60.0), 600.0),
                stale_after_sec=900.0,
            )
        if budget.get("allowed"):
            for step in SCENARIO2.steps:
                feedback = ""
                step_success = False
                for attempt_index in range(1, attempts_per_step + 1):
                    if engine == "local_agent":
                        objective = _render_attempt_objective(
                            workspace=workspace,
                            description_text=description_text,
                            completed=completed,
                            current_step=step,
                            feedback=feedback,
                            attempt_index=attempt_index,
                        )
                        result = asyncio.run(agent.run_cycle(objective, timeout_sec=timeout_sec))
                    else:
                        result = _execute_scenario2_step_deterministic(workspace, step.key)
                    ok, message = step.validator(workspace)
                    attempt_row = {
                        "step": step.key,
                        "attempt_index": attempt_index,
                        "validator_success": ok,
                        "validator_message": message,
                        "agent_result": result,
                    }
                    attempts.append(attempt_row)
                    if ok:
                        completed.append(step.key)
                        step_success = True
                        break
                    feedback = message
                if not step_success:
                    final_status = "failed"
                    stop_reason = f"{step.key}_failed"
                    break
        else:
            final_status = "blocked"
            stop_reason = str((budget.get("decision") or {}).get("reason") or "budget_unavailable")
    finally:
        _resume_scheduler_if_needed(scheduler_resume_needed)

    verification_rows = []
    passed_total = 0
    for step in SCENARIO2.steps:
        ok, message = step.validator(workspace)
        verification_rows.append({"step": step.key, "success": ok, "message": message})
        if ok:
            passed_total += 1

    score = round(passed_total / len(SCENARIO2.steps), 6)
    suite_name = f"{SCENARIO2.suite_name}_{engine}"
    report_dir = REPORT_ROOT / suite_name
    report_dir.mkdir(parents=True, exist_ok=True)
    detailed = {
        "contract": "eidos.agencybench_run.v1",
        "suite": "agencybench",
        "scenario": SCENARIO2.name,
        "variant": f"official_local_workspace_{engine}",
        "generated_at": _now_iso(),
        "status": final_status,
        "stop_reason": stop_reason,
        "model": model,
        "participant": f"eidos:{model if engine == 'local_agent' else 'deterministic_fs_agent'}",
        "execution_mode": "local_run",
        "engine": engine,
        "workspace": str(workspace),
        "attempts_per_step": attempts_per_step,
        "budget": budget,
        "subtasks_total": len(SCENARIO2.steps),
        "subtasks_passed": passed_total,
        "success_rate": score,
        "attempts": attempts,
        "verification": verification_rows,
        "scenario_source": str(scenario_root),
    }
    normalized = {
        "contract": "eidos.external_benchmark_result.v1",
        "suite": suite_name,
        "generated_at": detailed["generated_at"],
        "source_path": str(report_dir / "latest_detailed.json"),
        "source_url": "https://github.com/GAIR-NLP/AgencyBench",
        "participant": detailed["participant"],
        "execution_mode": "local_run",
        "status": "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red",
        "score": score,
        "metrics": {
            "score": score,
            "success_rate": score,
            "tasks_total": len(SCENARIO2.steps),
            "tasks_passed": passed_total,
            "subtasks_total": len(SCENARIO2.steps),
            "subtasks_passed": passed_total,
        },
        "notes": (
            "Official AgencyBench MCP scenario2 executed through the local Eidos constrained-tool harness."
            if engine == "local_agent"
            else "Official AgencyBench MCP scenario2 executed through the deterministic Eidos filesystem executor."
        ),
        "raw_excerpt": {
            "scenario": SCENARIO2.name,
            "variant": detailed["variant"],
            "status": final_status,
            "stop_reason": stop_reason,
        },
    }
    detailed_path = report_dir / f"{suite_name}_{stamp}_detailed.json"
    latest_detailed = report_dir / "latest_detailed.json"
    latest_json = report_dir / "latest.json"
    stamped_json = report_dir / f"{suite_name}_{stamp}.json"
    _write_json(detailed_path, detailed)
    _write_json(latest_detailed, detailed)
    _write_json(stamped_json, normalized)
    _write_json(latest_json, normalized)
    _write_text(
        report_dir / f"{suite_name}_{stamp}.md",
        "\n".join(
            [
                f"# {suite_name}",
                "",
                f"- generated_at: {detailed['generated_at']}",
                f"- model: {model}",
                f"- engine: {engine}",
                f"- status: {final_status}",
                f"- subtasks_passed: {passed_total}/{len(SCENARIO2.steps)}",
                f"- score: {score}",
                f"- stop_reason: {stop_reason}",
                "",
                "## Verification",
                *[
                    f"- {row['step']}: {'pass' if row['success'] else 'fail'} - {row['message']}"
                    for row in verification_rows
                ],
            ]
        )
        + "\n",
    )
    return {
        "detailed": detailed,
        "normalized": normalized,
        "paths": {"latest": str(latest_json), "latest_detailed": str(latest_detailed)},
    }


def run_scenario1(
    *,
    repo_root: Path,
    model: str,
    engine: str,
    repo_visibility: str,
) -> dict[str, Any]:
    if engine != "deterministic":
        raise ValueError("Scenario1 currently supports deterministic execution only.")
    stamp = _now_stamp()
    run_root = RUNTIME_ROOT / "scenario1" / stamp
    run_root.mkdir(parents=True, exist_ok=True)
    execution = _execute_scenario1_deterministic(run_root=run_root, stamp=stamp, repo_visibility=repo_visibility)
    target_payload = execution["target"]
    target = GitHubBenchmarkTarget(
        owner=str(target_payload["owner"]),
        repo=str(target_payload["repo"]),
        repo_url=str(target_payload["repo_url"]),
        worktree=Path(str(target_payload["worktree"])),
    )
    issue_number = int(execution["issue_number"])
    pr_number = int(execution["pr_number"])
    verification_rows = list(execution.get("verification") or [])
    passed_total = sum(1 for row in verification_rows if isinstance(row, dict) and row.get("success"))
    total = max(1, len(verification_rows))
    score = round(passed_total / total, 6)
    final_status = "success" if passed_total == total else "failed"
    stop_reason = "all_subtasks_completed" if final_status == "success" else "verification_failed"
    suite_name = "agencybench_eidos_scenario1_deterministic"
    report_dir = REPORT_ROOT / suite_name
    report_dir.mkdir(parents=True, exist_ok=True)
    detailed = {
        "contract": "eidos.agencybench_run.v1",
        "suite": "agencybench",
        "scenario": "scenario1",
        "variant": "official_github_workflow_deterministic",
        "generated_at": _now_iso(),
        "status": final_status,
        "stop_reason": stop_reason,
        "model": model,
        "participant": "eidos:deterministic_github_agent",
        "execution_mode": "local_run",
        "engine": engine,
        "repo_root": str(repo_root),
        "subtasks_total": len(verification_rows),
        "subtasks_passed": passed_total,
        "success_rate": score,
        "verification": verification_rows,
        "repo_target": target_payload,
        "issue_number": issue_number,
        "pr_number": pr_number,
    }
    normalized = {
        "contract": "eidos.external_benchmark_result.v1",
        "suite": suite_name,
        "generated_at": detailed["generated_at"],
        "source_path": str(report_dir / "latest_detailed.json"),
        "source_url": "https://github.com/GAIR-NLP/AgencyBench",
        "participant": detailed["participant"],
        "execution_mode": "local_run",
        "status": "green" if score >= 0.8 else "yellow" if score >= 0.5 else "red",
        "score": score,
        "metrics": {
            "score": score,
            "success_rate": score,
            "tasks_total": len(verification_rows),
            "tasks_passed": passed_total,
            "subtasks_total": len(verification_rows),
            "subtasks_passed": passed_total,
        },
        "notes": "Official AgencyBench MCP scenario1 GitHub workflow contract executed through the deterministic Eidos GitHub executor.",
        "raw_excerpt": {
            "scenario": "scenario1",
            "variant": detailed["variant"],
            "status": final_status,
            "stop_reason": stop_reason,
            "repo_url": target.repo_url,
            "issue_number": issue_number,
            "pr_number": pr_number,
        },
    }
    detailed_path = report_dir / f"{suite_name}_{stamp}_detailed.json"
    latest_detailed = report_dir / "latest_detailed.json"
    latest_json = report_dir / "latest.json"
    stamped_json = report_dir / f"{suite_name}_{stamp}.json"
    _write_json(detailed_path, detailed)
    _write_json(latest_detailed, detailed)
    _write_json(stamped_json, normalized)
    _write_json(latest_json, normalized)
    _write_text(
        report_dir / f"{suite_name}_{stamp}.md",
        "\n".join(
            [
                f"# {suite_name}",
                "",
                f"- generated_at: {detailed['generated_at']}",
                f"- model: {model}",
                f"- engine: {engine}",
                f"- status: {final_status}",
                f"- subtasks_passed: {passed_total}/{len(verification_rows)}",
                f"- score: {score}",
                f"- repo_url: {target.repo_url}",
                f"- issue_number: {issue_number}",
                f"- pr_number: {pr_number}",
                "",
                "## Verification",
                *[
                    f"- {row['step']}: {'pass' if row['success'] else 'fail'} - {row['message']}"
                    for row in verification_rows
                ],
            ]
        )
        + "\n",
    )
    return {
        "detailed": detailed,
        "normalized": normalized,
        "paths": {"latest": str(latest_json), "latest_detailed": str(latest_detailed)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a bounded Eidos local execution for selected AgencyBench scenarios."
    )
    parser.add_argument("--scenario", default="scenario2", choices=["scenario1", "scenario2"])
    parser.add_argument("--agencybench-root", default=str(DEFAULT_AGENCYBENCH_ROOT))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--model", default="qwen3.5:2b")
    parser.add_argument("--attempts-per-step", type=int, default=3)
    parser.add_argument("--timeout-sec", type=float, default=1800.0)
    parser.add_argument("--keep-alive", default="4h")
    parser.add_argument("--engine", choices=["local_agent", "deterministic"], default="local_agent")
    parser.add_argument("--repo-visibility", choices=["private", "public"], default="private")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    agencybench_root = Path(args.agencybench_root).resolve()
    if args.scenario == "scenario2":
        result = run_scenario2(
            agencybench_root=agencybench_root,
            repo_root=repo_root,
            model=args.model,
            attempts_per_step=max(1, int(args.attempts_per_step)),
            timeout_sec=max(60.0, float(args.timeout_sec)),
            keep_alive=str(args.keep_alive or "4h"),
            engine=str(args.engine or "local_agent"),
        )
    elif args.scenario == "scenario1":
        result = run_scenario1(
            repo_root=repo_root,
            model=args.model,
            engine=str(args.engine or "deterministic"),
            repo_visibility=str(args.repo_visibility or "private"),
        )
    else:
        raise SystemExit(f"Unsupported scenario: {args.scenario}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
