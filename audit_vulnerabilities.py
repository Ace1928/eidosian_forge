#!/usr/bin/env python3
"""Audit dependency vulnerabilities across the repo.

Usage:
  ./audit_vulnerabilities.py
  ./audit_vulnerabilities.py --fix
  ./audit_vulnerabilities.py --report docs/VULNERABILITY_REPORT.md

Example:
  ./audit_vulnerabilities.py --root /workspace/eidosian_forge --fix
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


REQ_GLOBS = (
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-prod.txt",
)


@dataclass(frozen=True)
class AuditTarget:
    kind: str
    path: Path


@dataclass(frozen=True)
class AuditResult:
    target: AuditTarget
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


def _log(prefix: str, message: str) -> None:
    print(f"{prefix} {message}")


def _find_targets(root: Path) -> list[AuditTarget]:
    targets: list[AuditTarget] = []
    for dirpath, _, filenames in os.walk(root):
        path = Path(dirpath)
        for filename in filenames:
            if filename in REQ_GLOBS:
                targets.append(AuditTarget("requirements", path / filename))
            elif filename == "package-lock.json":
                targets.append(AuditTarget("npm", path / filename))
            elif filename == "pyproject.toml":
                targets.append(AuditTarget("pyproject", path / filename))
    return targets


def _has_pinned_requirements(path: Path) -> bool:
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return False
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "==" in stripped or "@" in stripped:
            return True
    return False


def _run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )


def _audit_requirements(target: AuditTarget, fix: bool) -> AuditResult | None:
    pip_audit = shutil.which("pip-audit")
    if not pip_audit:
        _log("WARN", f"pip-audit not found; skipping {target.path}")
        return None
    command = [pip_audit, "-r", str(target.path), "--format", "json"]
    if fix:
        command.append("--fix")
    _log("INFO", f"Running: {' '.join(command)}")
    result = _run_command(command, cwd=target.path.parent)
    return AuditResult(target, command, result.returncode, result.stdout, result.stderr)


def _audit_npm(target: AuditTarget, fix: bool) -> AuditResult | None:
    npm = shutil.which("npm")
    if not npm:
        _log("WARN", f"npm not found; skipping {target.path}")
        return None
    command = [npm, "audit", "--json"]
    if fix:
        command.append("fix")
    _log("INFO", f"Running: {' '.join(command)}")
    result = _run_command(command, cwd=target.path.parent)
    return AuditResult(target, command, result.returncode, result.stdout, result.stderr)


def _note_pyproject(target: AuditTarget) -> None:
    _log(
        "WARN",
        f"pyproject.toml detected without a lockfile: {target.path}",
    )


def _format_header(text: str) -> str:
    return f"{text}\n{'=' * len(text)}\n"


def _json_loads(output: str) -> dict | list | None:
    cleaned = output.strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = min(
            (idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx != -1),
            default=-1,
        )
        if start == -1:
            return None
        try:
            return json.loads(cleaned[start:])
        except json.JSONDecodeError:
            return None


def _summarize_pip_audit(data: dict | list | None) -> list[str]:
    if not data or not isinstance(data, dict):
        return ["No JSON data returned by pip-audit."]
    dependencies = data.get("dependencies")
    if not dependencies:
        return ["No vulnerabilities reported."]
    lines = []
    for dependency in dependencies:
        name = dependency.get("name", "unknown")
        version = dependency.get("version", "unknown")
        vulns = dependency.get("vulns") or []
        for vuln in vulns:
            vuln_id = vuln.get("id", "unknown")
            fix_versions = vuln.get("fix_versions") or []
            fix_text = ", ".join(fix_versions) if fix_versions else "none"
            lines.append(
                f"- {name} {version}: {vuln_id} (fix versions: {fix_text})"
            )
    if not lines:
        return ["No vulnerabilities reported."]
    return lines


def _summarize_npm_audit(data: dict | list | None) -> list[str]:
    if not data or not isinstance(data, dict):
        return ["No JSON data returned by npm audit."]
    vulnerabilities = data.get("vulnerabilities", {})
    if not vulnerabilities:
        return ["No vulnerabilities reported."]
    lines = []
    for name, details in sorted(vulnerabilities.items()):
        severity = details.get("severity", "unknown")
        via = details.get("via", [])
        if isinstance(via, list):
            via_text = ", ".join(
                item.get("source", str(item)) if isinstance(item, dict) else str(item)
                for item in via
            )
        else:
            via_text = str(via)
        lines.append(f"- {name}: {severity} (via: {via_text})")
    return lines


def _write_report(
    report_path: Path,
    results: list[AuditResult],
    pyproject_warnings: list[AuditTarget],
    skipped_requirements: list[AuditTarget],
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        _format_header("Vulnerability Report"),
        f"Generated: {timestamp}",
        "",
        _format_header("Summary"),
    ]
    requirement_results = [r for r in results if r.target.kind == "requirements"]
    npm_results = [r for r in results if r.target.kind == "npm"]
    lines.extend(
        [
            f"- Requirements audits: {len(requirement_results)}",
            f"- npm audits: {len(npm_results)}",
            f"- pyproject.toml without lockfiles: {len(pyproject_warnings)}",
            f"- requirements without pins (skipped): {len(skipped_requirements)}",
            "",
            _format_header("Details"),
        ]
    )
    for result in results:
        lines.append(f"## {result.target.path}")
        lines.append(f"Command: {' '.join(result.command)}")
        lines.append(f"Return code: {result.returncode}")
        data = _json_loads(result.stdout)
        if result.target.kind == "requirements":
            lines.append("Findings:")
            lines.extend(_summarize_pip_audit(data))
        elif result.target.kind == "npm":
            lines.append("Findings:")
            lines.extend(_summarize_npm_audit(data))
        if result.stderr.strip():
            lines.append("stderr:")
            lines.append(result.stderr.strip())
        lines.append("")
    if pyproject_warnings:
        lines.append(_format_header("pyproject.toml without lockfiles"))
        lines.extend(f"- {target.path}" for target in pyproject_warnings)
        lines.append("")
    if skipped_requirements:
        lines.append(_format_header("requirements without pinned versions"))
        lines.extend(f"- {target.path}" for target in skipped_requirements)
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit repo dependencies with pip-audit and npm audit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  ./audit_vulnerabilities.py --root /workspace/eidosian_forge --fix"
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Root directory to scan (default: cwd).",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Apply fixes when supported by the audit tool.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Write a Markdown vulnerability report to this path.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    if not root.exists():
        _log("ERROR", f"Root path not found: {root}")
        return 2

    targets = _find_targets(root)
    if not targets:
        _log("WARN", f"No dependency files found under {root}")
        return 0

    failures = 0
    results: list[AuditResult] = []
    pyproject_warnings: list[AuditTarget] = []
    skipped_requirements: list[AuditTarget] = []
    for target in sorted(targets, key=lambda item: (item.kind, str(item.path))):
        if target.kind == "requirements":
            if not _has_pinned_requirements(target.path):
                _log("WARN", f"No pinned requirements found; skipping {target.path}")
                skipped_requirements.append(target)
                continue
            result = _audit_requirements(target, args.fix)
            if result is not None:
                results.append(result)
                failures += 1 if result.returncode != 0 else 0
        elif target.kind == "npm":
            result = _audit_npm(target, args.fix)
            if result is not None:
                results.append(result)
                failures += 1 if result.returncode != 0 else 0
        elif target.kind == "pyproject":
            _note_pyproject(target)
            pyproject_warnings.append(target)

    if args.report:
        _write_report(args.report, results, pyproject_warnings, skipped_requirements)
        _log("INFO", f"Wrote report to {args.report}")

    if failures:
        _log("ERROR", f"Audit completed with {failures} failing command(s).")
        return 1

    _log("INFO", "Audit completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
