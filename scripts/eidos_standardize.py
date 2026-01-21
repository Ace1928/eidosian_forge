#!/usr/bin/env python3
"""
Eidosian Standardization Script
-------------------------------
Automates the creation of standard documentation and packaging scaffolds across
top-level directories in the Eidosian Forge.

Usage:
    python scripts/eidos_standardize.py --mode report
    python scripts/eidos_standardize.py --mode docs
    python scripts/eidos_standardize.py --mode packaging --targets agent_forge eidos_mcp
    python scripts/eidos_standardize.py --mode full
"""

import argparse
import sys
import os
import textwrap
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple

# Add root to sys.path to import global_info
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

try:
    import global_info
except ImportError:
    from datetime import date

    class _FallbackInfo:
        @staticmethod
        def get_version() -> str:
            return "unknown"

        date = date

    global_info = _FallbackInfo()

# Configuration
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    ".vscode",
    ".pytest_cache",
    ".venv_tools",
    "node_modules",
}

DATA_DIRS = {
    ".github",
    "audit_data",
    "data",
    "docs",
    "lib",
    "logs",
    "memory_db",
    "reports",
    "requirements",
    "scripts",
    "tests",
}

CONTAINER_DIRS = {"projects"}

STANDARD_FILES = ["README.md", "CURRENT_STATE.md", "GOALS.md", "TODO.md"]
MAX_SCAN_FILES = 50000


def _iter_dirs(root: Path, targets: Optional[List[str]] = None) -> List[Path]:
    """Return top-level directories to process."""
    if targets:
        resolved = []
        for name in targets:
            path = root / name
            if path.exists() and path.is_dir():
                resolved.append(path)
        return sorted(resolved)
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name not in SKIP_DIRS])


def _count_files(path: Path, suffix: Optional[str] = None) -> Tuple[int, bool]:
    """Count files with optional suffix; cap at MAX_SCAN_FILES."""
    count = 0
    capped = False
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if suffix and not fname.endswith(suffix):
                continue
            count += 1
            if count >= MAX_SCAN_FILES:
                capped = True
                return count, capped
    return count, capped


def _has_src_package(path: Path) -> Tuple[bool, List[str]]:
    src_dir = path / "src"
    if not src_dir.exists():
        return False, []
    names = []
    for sub in src_dir.iterdir():
        if sub.is_dir() and (sub / "__init__.py").exists():
            names.append(sub.name)
    return bool(names), sorted(names)


def _infer_kind(path: Path) -> str:
    if path.name in DATA_DIRS:
        return "data"
    if path.name in CONTAINER_DIRS:
        return "container"
    pyproject = (path / "pyproject.toml").exists()
    setup_py = (path / "setup.py").exists()
    src_pkg, _ = _has_src_package(path)
    python_files, _ = _count_files(path, suffix=".py")
    if pyproject or setup_py or src_pkg or python_files:
        return "python"
    return "misc"

def generate_readme(module_name: str, report: Dict[str, Any]) -> str:
    """Generate a standardized README content."""
    title = module_name.replace("_", " ").title()
    if module_name == "eidos_brain":
        title = "Eidos Brain"
    total_files = report["counts"]["total_files"]
    python_files = report["counts"]["python_files"]
    total_note = " (capped)" if report["counts"]["total_capped"] else ""
    python_note = " (capped)" if report["counts"]["python_capped"] else ""
    tests = "present" if report["packaging"]["tests_dir"] else "missing"
    pyproject = "present" if report["packaging"]["pyproject"] else "missing"
    return f"""# {title}

**Eidosian Module**: `{module_name}`

## Overview
- Type: {report["kind"]}
- Total files: {total_files}{total_note}
- Python files: {python_files}{python_note}
- Tests: {tests}
- Packaging (pyproject): {pyproject}

## Structure
- Root: `{module_name}/`
- Source: `{module_name}/src/` (if present)
- Tests: `{module_name}/tests/` (if present)
"""

def generate_current_state(module_name: str, report: Dict[str, Any]) -> str:
    """Generate CURRENT_STATE.md content."""
    status = "Needs standardization" if report["gaps"] else "Baseline ok"
    gaps = report["gaps"] or ["None detected by automated scan."]
    gaps_text = "\n".join([f"- {gap}" for gap in gaps])
    return f"""# Current State: {module_name}

Date: {global_info.date.today().strftime('%Y-%m-%d')}
Status: {status}

## Metrics
- Total files: {report["counts"]["total_files"]}{" (capped)" if report["counts"]["total_capped"] else ""}
- Python files: {report["counts"]["python_files"]}{" (capped)" if report["counts"]["python_capped"] else ""}
- Tests directory: {"present" if report["packaging"]["tests_dir"] else "missing"}

## Packaging Snapshot
- pyproject.toml: {"present" if report["packaging"]["pyproject"] else "missing"}
- setup.py: {"present" if report["packaging"]["setup_py"] else "missing"}
- root __init__.py: {"present" if report["packaging"]["root_init"] else "missing"}
- src packages: {", ".join(report["packaging"]["src_packages"]) or "none"}

## Gaps
{gaps_text}
"""

def generate_goals(module_name: str, report: Dict[str, Any]) -> str:
    """Generate GOALS.md content."""
    goals = []
    if report["kind"] == "python":
        if not report["packaging"]["pyproject"]:
            goals.append("Add a pyproject.toml for build metadata.")
        if not report["packaging"]["tests_dir"]:
            goals.append("Add or expand tests for core functionality.")
        if not report["packaging"]["root_init"]:
            goals.append("Add root __init__.py for import path bridging.")
    elif report["kind"] == "data":
        goals.append("Document data provenance and retention.")
    elif report["kind"] == "container":
        goals.append("Catalog subprojects and ownership.")
    if not goals:
        goals.append("Maintain current structure and keep dependencies updated.")
    goals_text = "\n".join([f"- [ ] {goal}" for goal in goals])
    return f"""# Goals: {module_name}

{goals_text}
"""

def generate_todo(module_name: str, report: Dict[str, Any]) -> str:
    """Generate TODO.md content."""
    tasks = []
    if report["kind"] == "python":
        if not report["packaging"]["pyproject"]:
            tasks.append("Add pyproject.toml with build metadata.")
        if not report["packaging"]["tests_dir"]:
            tasks.append("Add tests for core paths.")
        if report["packaging"]["src_packages"] and module_name not in report["packaging"]["src_packages"]:
            tasks.append("Add src bridge package matching directory name.")
    elif report["kind"] == "data":
        tasks.append("Document datasets, sources, and update cadence.")
    elif report["kind"] == "container":
        tasks.append("Enumerate subprojects and their test commands.")
    if not tasks:
        tasks.append("Review dependencies and update changelog.")
    tasks_text = "\n".join([f"- [ ] {task}" for task in tasks])
    return f"""# TODO: {module_name}

{tasks_text}
"""


def _build_report(path: Path) -> Dict[str, Any]:
    kind = _infer_kind(path)
    total_files, total_capped = _count_files(path)
    python_files, python_capped = _count_files(path, suffix=".py")
    src_pkg, src_names = _has_src_package(path)
    report = {
        "name": path.name,
        "kind": kind,
        "counts": {
            "total_files": total_files,
            "python_files": python_files,
            "total_capped": total_capped,
            "python_capped": python_capped,
        },
        "packaging": {
            "pyproject": (path / "pyproject.toml").exists(),
            "setup_py": (path / "setup.py").exists(),
            "root_init": (path / "__init__.py").exists(),
            "src_packages": src_names,
            "tests_dir": (path / "tests").exists(),
        },
        "gaps": [],
    }
    if kind == "python":
        if not report["packaging"]["pyproject"]:
            report["gaps"].append("Missing pyproject.toml.")
        if not report["packaging"]["root_init"]:
            report["gaps"].append("Missing root __init__.py for import bridging.")
        if src_pkg and path.name not in src_names:
            report["gaps"].append(
                f"src package names do not include {path.name}."
            )
        if not report["packaging"]["tests_dir"]:
            report["gaps"].append("Missing tests/ directory.")
    return report

def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def process_module(module_path: Path, mode: str) -> Dict[str, Any]:
    """Check and create standard files for a module."""
    report = _build_report(module_path)
    print(f"🔍 Processing {module_path.name} ({report['kind']})...")

    if mode in {"docs", "full"}:
        generators = {
            "README.md": generate_readme,
            "CURRENT_STATE.md": generate_current_state,
            "GOALS.md": generate_goals,
            "TODO.md": generate_todo,
        }

        for filename, generator in generators.items():
            file_path = module_path / filename
            if not file_path.exists():
                print(f"  ➕ Creating {filename}")
                content = generator(module_path.name, report)
                try:
                    _write_file(file_path, content)
                except Exception as exc:
                    print(f"  ❌ Error writing {filename}: {exc}")

    if mode in {"packaging", "full"} and report["kind"] == "python":
        _ensure_packaging(module_path, report)

    return report


def _ensure_packaging(module_path: Path, report: Dict[str, Any]) -> None:
    """Ensure packaging scaffolds exist without moving files."""
    if not report["packaging"]["root_init"]:
        init_path = module_path / "__init__.py"
        content = textwrap.dedent("""\"\"\"Root package bridge for Eidosian layout.\"\"\"
from pkgutil import extend_path
from pathlib import Path
__path__ = extend_path(__path__, __name__)
_root = Path(__file__).resolve().parent
_src = _root / "src" / __name__
if _src.exists():
    __path__.append(str(_src))
_nested = _root / __name__
if _nested.exists():
    __path__.append(str(_nested))
""").lstrip()
        _write_file(init_path, content)
        print("  ➕ Added root __init__.py bridge")

    if not report["packaging"]["pyproject"]:
        pyproject = module_path / "pyproject.toml"
        package_names = report["packaging"]["src_packages"] or [module_path.name]
        packages_line = ", ".join([f"\"src/{name}\"" for name in package_names])
        content = textwrap.dedent(f"""[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{module_path.name}"
version = "0.0.1"
description = "Eidosian module: {module_path.name}"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
authors = [
    {{ name = "Lloyd Handyside", email = "ace1928@gmail.com" }},
    {{ name = "Eidos", email = "syntheticeidos@gmail.com" }},
]

[tool.hatch.build.targets.wheel]
packages = [{packages_line}]

[tool.pytest.ini_options]
minversion = "6.0"
addopts = "-ra -q"
testpaths = ["tests"]
""").lstrip()
        _write_file(pyproject, content)
        print("  ➕ Added pyproject.toml")

    src_dir = module_path / "src" / module_path.name
    if not src_dir.exists():
        src_dir.mkdir(parents=True, exist_ok=True)
    init_path = src_dir / "__init__.py"
    if not init_path.exists():
        bridge_targets = report["packaging"]["src_packages"]
        imports = []
        if bridge_targets:
            for name in bridge_targets:
                imports.append(f"from {name} import *  # type: ignore")
        content = textwrap.dedent("""\"\"\"Bridge package for legacy layout.\"\"\"
""").lstrip()
        if imports:
            content += "\n".join(imports) + "\n"
        _write_file(init_path, content)
        print("  ➕ Added src bridge package")

def main():
    parser = argparse.ArgumentParser(description="Eidosian standardization tool")
    parser.add_argument(
        "--mode",
        choices=["report", "docs", "packaging", "full"],
        default="report",
        help="Standardization mode",
    )
    parser.add_argument(
        "--targets",
        nargs="*",
        help="Optional list of top-level directory names to process",
    )
    args = parser.parse_args()

    print(f"🚀 Eidosian Standardization Tool v{global_info.get_version()}")
    print(f"📂 Root: {root_dir}")
    print(f"mode={args.mode}")

    modules = _iter_dirs(root_dir, targets=args.targets)
    print(f"found {len(modules)} directories.")

    reports = []
    for module in modules:
        reports.append(process_module(module, args.mode))

    if args.mode == "report":
        missing = [r for r in reports if r["gaps"]]
        print(f"gapped={len(missing)}")

    print("\n✅ Standardization complete.")

if __name__ == "__main__":
    main()
