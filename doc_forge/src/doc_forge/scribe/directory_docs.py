from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

README_NAMES = ("README.md", "README")
ROUTE_RE = re.compile(r"""@(?:app|router|[A-Za-z_]\w*)\.(get|post|put|patch|delete)\(\s*["']([^"']+)["']""")

DEFAULT_POLICY: dict[str, Any] = {
    "contract": "eidos.documentation_policy.v1",
    "documented_prefixes": [
        ".github",
        "agent_forge",
        "article_forge",
        "audit_forge",
        "benchmarks",
        "bin",
        "cfg",
        "code_forge",
        "computer_control_forge",
        "config",
        "crawl_forge",
        "diagnostics_forge",
        "doc_forge",
        "docs",
        "eidos_mcp",
        "erais_forge",
        "figlet_forge",
        "file_forge",
        "game_forge",
        "gis_forge",
        "glyph_forge",
        "knowledge_forge",
        "lib",
        "llm_forge",
        "lyrics_forge",
        "memory_forge",
        "metadata_forge",
        "moltbook_forge",
        "narrative_forge",
        "ollama_forge",
        "prompt_forge",
        "refactor_forge",
        "repo_forge",
        "requirements",
        "scripts",
        "shell",
        "skills",
        "sms_forge",
        "terminal_forge",
        "test_forge",
        "type_forge",
        "version_forge",
        "viz_forge",
        "web_interface_forge",
        "word_forge",
    ],
    "excluded_prefixes": [
        ".eidos_chrome_profile",
        ".gemini",
        ".graphrag_index/cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tmp_test",
        ".vscode",
        "Backups",
        "backups",
        "build",
        "cache",
        "data",
        "doc_forge/final_docs",
        "doc_forge/runtime",
        "doc_forge/staging",
        "eidosian_venv",
        "logs",
        "reports",
        "state",
    ],
    "excluded_segments": [
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "node_modules",
        ".git",
    ],
    "path_overrides": {
        "doc_forge/src/doc_forge": {
            "summary": "Primary Doc Forge Python package for repository-scale documentation extraction, generation, judging, and service orchestration.",
            "why": "It centralizes documentation production logic so documentation can be treated as a first-class maintained system instead of ad hoc markdown.",
            "next_steps": [
                "Expand the documentation contract deeper across package subdirectories.",
                "Keep API and service claims synchronized with test coverage.",
            ],
        },
        "doc_forge/src/doc_forge/scribe": {
            "summary": "Production document-processing pipeline: extract source context, generate docs, judge quality, persist state, and expose service APIs.",
            "why": "This is the operational heart of Doc Forge and the main integration point for living documentation automation.",
            "next_steps": [
                "Extend route coverage for documentation inventory and managed README generation.",
                "Benchmark generation quality and latency per directory class.",
            ],
        },
        "web_interface_forge/src/web_interface_forge/dashboard": {
            "summary": "Atlas dashboard backend and templates for runtime control, graph exploration, and operator-facing forge observability.",
            "why": "It gives the forge an operator plane instead of requiring shell-only control and log inspection.",
            "next_steps": [
                "Expose documentation coverage and diff workflows directly in Atlas.",
                "Continue converging service, scheduler, and documentation control in one surface.",
            ],
        },
        "agent_forge/src/agent_forge/consciousness": {
            "summary": "Consciousness runtime, metrics, perturbation, and benchmark substrate used by Eidos runtime status and experiments.",
            "why": "This package supplies the operational continuity, bridge state, and evaluative signals that the rest of the control plane depends on.",
            "next_steps": [
                "Keep bridge and memory defaults aligned with the canonical vector-native stores.",
                "Expand explicit evidence lineage and contradiction analysis in the graph-facing outputs.",
            ],
        },
        "lib/eidosian_runtime": {
            "summary": "Shared runtime coordination helpers for scheduler, capability registry, and service-state governance.",
            "why": "It keeps platform/runtime checks out of leaf modules and gives Atlas, boot scripts, and scheduler one shared contract.",
            "next_steps": [
                "Continue moving platform checks out of leaf modules into capability-driven paths.",
                "Expose longer-lived runtime history and policy state through the same contract.",
            ],
        },
    },
}


@dataclass(slots=True)
class DirectoryDocRecord:
    path: str
    readme_path: str
    required: bool
    has_readme: bool
    tracked_files: int
    child_directories: list[str]
    prominent_files: list[str]
    python_modules: list[str]
    api_routes: list[str]
    tests_present: bool
    summary: str
    why: str
    strengths: list[str]
    weaknesses: list[str]
    next_steps: list[str]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_policy(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "cfg" / "documentation_policy.json"
    if not path.exists():
        return DEFAULT_POLICY
    payload = json.loads(path.read_text(encoding="utf-8"))
    merged = dict(DEFAULT_POLICY)
    merged.update(payload)
    merged["path_overrides"] = {**DEFAULT_POLICY.get("path_overrides", {}), **payload.get("path_overrides", {})}
    return merged


def _git_ls_files(repo_root: Path, selected_paths: set[str] | None = None) -> list[str]:
    command = ["git", "-C", str(repo_root), "ls-files", "-z"]
    selected = sorted({p for p in (selected_paths or set()) if p})
    if selected:
        command.extend(["--", *selected])
    raw = subprocess.check_output(command)
    return [p for p in raw.decode("utf-8", "replace").split("\0") if p]


def _iter_parent_dirs(rel_file: str):
    parent = Path(rel_file).parent
    while str(parent) not in {"", "."}:
        yield parent.as_posix()
        parent = parent.parent


def _is_excluded(rel_dir: str, policy: dict[str, Any]) -> bool:
    parts = Path(rel_dir).parts
    excluded_segments = set(policy.get("excluded_segments", []))
    if any(part in excluded_segments or part.endswith(".egg-info") for part in parts):
        return True
    for prefix in policy.get("excluded_prefixes", []):
        if rel_dir == prefix or rel_dir.startswith(prefix + "/"):
            return True
    return False


def _is_documented_scope(rel_dir: str, policy: dict[str, Any]) -> bool:
    for prefix in policy.get("documented_prefixes", []):
        if rel_dir == prefix or rel_dir.startswith(prefix + "/"):
            return True
    return False


def _category(rel_dir: str) -> str:
    base = Path(rel_dir).name
    if base in {"src", "core", "services", "utils", "config", "cfg", "cli", "dashboard", "autonomy"}:
        return base
    if base in {"docs", "tests", "scripts", "benchmarks", "templates", "examples"}:
        return base
    top = rel_dir.split("/", 1)[0]
    if top.endswith("_forge"):
        return "forge"
    if top == "lib":
        return "library"
    return "directory"


def _matches_selected_prefix(rel_dir: str, selected_paths: set[str]) -> bool:
    if not selected_paths:
        return True
    for prefix in selected_paths:
        if rel_dir == prefix or rel_dir.startswith(prefix + "/") or prefix.startswith(rel_dir + "/"):
            return True
    return False


def _relative_link(from_dir: str, to_path: str) -> str:
    return os.path.relpath(to_path, start=from_dir).replace(os.sep, "/")


def _reference_links(repo_root: Path, rel_dir: str) -> list[tuple[str, str, str]]:
    refs: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    current = Path(rel_dir)
    for ancestor in current.parents:
        ancestor_rel = ancestor.as_posix()
        if ancestor_rel in {"", "."}:
            continue
        readme = repo_root / ancestor_rel / "README.md"
        if readme.exists():
            target = f"{ancestor_rel}/README.md"
            refs.append(("Parent README", target, _relative_link(rel_dir, target)))
            seen.add(target)
            break
    forge_root = rel_dir.split("/", 1)[0]
    forge_target = f"{forge_root}/README.md"
    if forge_target not in seen and (repo_root / forge_target).exists():
        refs.append(("Forge README", forge_target, _relative_link(rel_dir, forge_target)))
    return refs


def inventory_directories(
    repo_root: Path,
    policy: dict[str, Any] | None = None,
    selected_paths: set[str] | None = None,
) -> list[DirectoryDocRecord]:
    policy = policy or load_policy(repo_root)
    selected_paths = {p.strip().strip("/") for p in (selected_paths or set()) if p.strip()}
    tracked_files = _git_ls_files(repo_root, selected_paths=selected_paths or None)
    dir_to_files: dict[str, list[str]] = {}
    dir_children: dict[str, set[str]] = {}
    dirs: set[str] = set()
    for rel in tracked_files:
        for parent in _iter_parent_dirs(rel):
            if _is_excluded(parent, policy) or not _is_documented_scope(parent, policy):
                continue
            if not _matches_selected_prefix(parent, selected_paths):
                continue
            dirs.add(parent)
            dir_to_files.setdefault(parent, []).append(rel)
            pp = Path(parent)
            gp = pp.parent.as_posix()
            if (
                gp != "."
                and _is_documented_scope(gp, policy)
                and not _is_excluded(gp, policy)
                and _matches_selected_prefix(gp, selected_paths)
            ):
                dir_children.setdefault(gp, set()).add(parent)
    records: list[DirectoryDocRecord] = []
    for rel_dir in sorted(dirs):
        abs_dir = repo_root / rel_dir
        readme_path = f"{rel_dir}/README.md"
        has_readme = (abs_dir / "README.md").exists() or (abs_dir / "README").exists()
        files = sorted(dir_to_files.get(rel_dir, []))
        child_dirs = sorted([Path(p).name for p in dir_children.get(rel_dir, set())])
        local_files = []
        for rel_file in files:
            if Path(rel_file).parent.as_posix() == rel_dir:
                local_files.append(Path(rel_file).name)
        prominent_files = sorted(local_files)[:8]
        python_modules = sorted(
            [
                Path(f).stem
                for f in files
                if f.endswith(".py") and Path(f).parent.as_posix() == rel_dir and Path(f).name != "__init__.py"
            ]
        )[:10]
        api_routes = []
        for rel_file in files:
            if not rel_file.endswith(".py"):
                continue
            fp = repo_root / rel_file
            try:
                text = fp.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for method, route in ROUTE_RE.findall(text):
                api_routes.append(f"{method.upper()} {route}")
        api_routes = sorted(dict.fromkeys(api_routes))[:20]
        tests_present = any("/tests/" in f or Path(f).name.startswith("test_") for f in files)
        override = policy.get("path_overrides", {}).get(rel_dir, {})
        summary = (
            override.get("summary")
            or f"Managed directory documentation for `{rel_dir}`, generated from tracked files and directory structure."
        )
        why = (
            override.get("why")
            or f"This directory exists to hold the `{_category(rel_dir)}` surface for `{rel_dir.split('/', 1)[0]}` and keep that responsibility separate from adjacent forge concerns."
        )
        strengths = []
        if tests_present:
            strengths.append("A directly associated test surface is present in or below this directory.")
        if api_routes:
            strengths.append(
                "The directory contains detected HTTP/API route definitions that can be referenced programmatically."
            )
        if python_modules:
            strengths.append("The directory exposes importable Python modules rather than only opaque assets.")
        if child_dirs:
            strengths.append("Responsibility is split into child directories instead of one flat file heap.")
        weaknesses = []
        if not has_readme:
            weaknesses.append("The directory had no local README before this managed documentation pass.")
        if not tests_present:
            weaknesses.append(
                "No directly associated test coverage was detected under the tracked file set for this directory."
            )
        if not api_routes and _category(rel_dir) in {"services", "dashboard", "api", "cli"}:
            weaknesses.append(
                "No route or explicit API entrypoint was detected automatically; verify interface coverage manually."
            )
        if not child_dirs and len(files) > 12:
            weaknesses.append(
                "The directory has many tracked files but no child-directory decomposition, which may make ownership blur over time."
            )
        next_steps = list(override.get("next_steps") or [])
        if not next_steps:
            if not tests_present:
                next_steps.append(
                    "Add focused tests or point this directory explicitly at its validating test surface."
                )
            if not api_routes and _category(rel_dir) in {"services", "dashboard", "api", "cli"}:
                next_steps.append("Document or expose the public interface contract more explicitly.")
            if has_readme:
                next_steps.append(
                    "Keep this README synchronized with code and test changes through the managed documentation toolchain."
                )
            else:
                next_steps.append("Adopt and retain the managed README generated for this directory.")
        records.append(
            DirectoryDocRecord(
                path=rel_dir,
                readme_path=readme_path,
                required=True,
                has_readme=has_readme,
                tracked_files=len(files),
                child_directories=child_dirs,
                prominent_files=prominent_files,
                python_modules=python_modules,
                api_routes=api_routes,
                tests_present=tests_present,
                summary=summary,
                why=why,
                strengths=strengths,
                weaknesses=weaknesses,
                next_steps=next_steps,
            )
        )
    return records


def inventory_summary(
    repo_root: Path,
    policy: dict[str, Any] | None = None,
    selected_paths: set[str] | None = None,
) -> dict[str, Any]:
    policy = policy or load_policy(repo_root)
    records = inventory_directories(repo_root, policy, selected_paths=selected_paths)
    missing = [r.path for r in records if not r.has_readme]
    return {
        "contract": "eidos.documentation_inventory.v1",
        "generated_at": _now(),
        "repo_root": str(repo_root.resolve()),
        "required_directory_count": len(records),
        "missing_readme_count": len(missing),
        "missing_readmes": missing,
        "records": [asdict(r) for r in records],
    }


def record_map(
    repo_root: Path,
    policy: dict[str, Any] | None = None,
    selected_paths: set[str] | None = None,
) -> dict[str, DirectoryDocRecord]:
    return {record.path: record for record in inventory_directories(repo_root, policy, selected_paths=selected_paths)}


def render_directory_readme(
    repo_root: Path,
    rel_dir: str,
    policy: dict[str, Any] | None = None,
    records: dict[str, DirectoryDocRecord] | None = None,
) -> str:
    policy = policy or load_policy(repo_root)
    records = records or record_map(repo_root, policy, selected_paths={rel_dir})
    record = records.get(rel_dir)
    if record is None:
        raise FileNotFoundError(f"Directory not in managed documentation scope: {rel_dir}")
    lines = [
        f"# `{record.path}`",
        "",
        "<!-- EIDOS:DOCSYS:START -->",
        "- Contract: `eidos.directory_doc.v1`",
        f"- Generated: `{_now()}`",
        f"- Path: `{record.path}`",
        "",
        "## What It Is",
        "",
        record.summary,
        "",
        "## Why It Exists",
        "",
        record.why,
        "",
        "## How It Works",
        "",
        f"- Tracked files in scope: `{record.tracked_files}`",
        f"- Child directories: `{len(record.child_directories)}`",
        f"- Tests detected: `{record.tests_present}`",
    ]
    if record.python_modules:
        lines.append(f"- Python modules: `{', '.join(record.python_modules)}`")
    if record.api_routes:
        lines.append(f"- Detected API routes: `{'; '.join(record.api_routes[:8])}`")
    lines.extend(["", "## Contents", ""])
    if record.child_directories:
        for name in record.child_directories:
            child_readme = repo_root / record.path / name / "README.md"
            if child_readme.exists():
                lines.append(f"- [`{name}`](./{name}/README.md)")
            else:
                lines.append(f"- `{name}`")
    else:
        lines.append("- No managed child directories detected.")
    if record.prominent_files:
        lines.extend(["", "## Prominent Files", ""])
        for name in record.prominent_files:
            lines.append(f"- [`{name}`](./{name})")
    if record.api_routes:
        lines.extend(["", "## API Surface", ""])
        for route in record.api_routes:
            lines.append(f"- `{route}`")
    lines.extend(["", "## Strengths", ""])
    if record.strengths:
        for item in record.strengths:
            lines.append(f"- {item}")
    else:
        lines.append("- The directory is at least structurally documented through the managed documentation contract.")
    lines.extend(["", "## Weaknesses / Risks", ""])
    if record.weaknesses:
        for item in record.weaknesses:
            lines.append(f"- {item}")
    else:
        lines.append(
            "- No structural documentation risks were detected automatically; functional review is still required for behavior-level claims."
        )
    lines.extend(["", "## Next Steps", ""])
    for item in record.next_steps:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## References",
            "",
        ]
    )
    references = _reference_links(repo_root, rel_dir)
    if references:
        for label, target, href in references:
            lines.append(f"- {label}: [`{target}`]({href})")
    else:
        lines.append("- No ancestor README reference was detected automatically.")
    lines.extend(
        [
            "",
            "## Accuracy Contract",
            "",
            "- This README is generated from tracked repository structure and conservative code scanning.",
            "- Behavior claims should be backed by tests, routes, or directly linked source files.",
            "<!-- EIDOS:DOCSYS:END -->",
            "",
        ]
    )
    return "\n".join(lines)


def readme_diff(
    repo_root: Path,
    rel_dir: str,
    policy: dict[str, Any] | None = None,
    records: dict[str, DirectoryDocRecord] | None = None,
) -> str:
    path = repo_root / rel_dir / "README.md"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    proposed = render_directory_readme(repo_root, rel_dir, policy, records=records)
    diff = difflib.unified_diff(
        existing.splitlines(),
        proposed.splitlines(),
        fromfile=f"{rel_dir}/README.md",
        tofile=f"{rel_dir}/README.md",
        lineterm="",
    )
    text = "\n".join(diff)
    return text + ("\n" if text else "")


def upsert_directory_readme(
    repo_root: Path,
    rel_dir: str,
    policy: dict[str, Any] | None = None,
    records: dict[str, DirectoryDocRecord] | None = None,
) -> dict[str, Any]:
    content = render_directory_readme(repo_root, rel_dir, policy, records=records)
    path = repo_root / rel_dir / "README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    prior = path.read_text(encoding="utf-8") if path.exists() else ""
    path.write_text(content, encoding="utf-8")
    return {
        "path": rel_dir,
        "readme_path": str(path.relative_to(repo_root)),
        "changed": prior != content,
        "created": not bool(prior),
        "content": content,
    }
