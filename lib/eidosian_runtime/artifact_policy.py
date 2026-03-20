from __future__ import annotations

import fnmatch
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List

_DEFAULT_POLICY_PATH = Path("cfg/runtime_artifact_policy.json")


def _pattern_roots(patterns: Iterable[str]) -> list[str]:
    roots: list[str] = []
    for pattern in patterns:
        text = str(pattern).strip().replace("\\", "/")
        if not text:
            continue
        root = text.split("/", 1)[0]
        if root and root not in roots:
            roots.append(root)
    return roots or ["."]


def _normalize_paths(paths: Iterable[str]) -> list[str]:
    return sorted({str(path).replace("\\", "/") for path in paths if str(path).strip()})


def load_runtime_artifact_policy(repo_root: str | Path, policy_path: str | Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    candidate = Path(policy_path) if policy_path is not None else root / _DEFAULT_POLICY_PATH
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    data = json.loads(candidate.read_text(encoding="utf-8"))
    tracked = _normalize_paths(data.get("tracked_generated_globs") or [])
    live = _normalize_paths(data.get("live_generated_globs") or [])
    return {
        "policy_path": str(candidate),
        "tracked_generated_globs": tracked,
        "live_generated_globs": live,
    }


def _git_ls_files(repo_root: Path, *, roots: Iterable[str] | None = None) -> list[str]:
    command = ["git", "-C", str(repo_root), "ls-files"]
    scoped_roots = [root for root in (roots or []) if str(root).strip()]
    if scoped_roots:
        command.extend(["--", *scoped_roots])
    proc = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def _glob_repo(repo_root: Path, patterns: Iterable[str]) -> list[str]:
    found: list[str] = []
    for pattern in patterns:
        if pattern.endswith("/**"):
            base = repo_root / pattern[:-3]
            if base.exists():
                found.append(str(base.relative_to(repo_root)).replace("\\", "/"))
            continue
        found.extend(str(path.relative_to(repo_root)).replace("\\", "/") for path in repo_root.glob(pattern))
    return _normalize_paths(found)


def _match_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def audit_runtime_artifacts(repo_root: str | Path, policy_path: str | Path | None = None) -> Dict[str, Any]:
    root = Path(repo_root).resolve()
    policy = load_runtime_artifact_policy(root, policy_path)
    tracked_patterns = policy["tracked_generated_globs"]
    live_patterns = policy["live_generated_globs"]
    tracked_roots = _pattern_roots(tracked_patterns)

    tracked_files = _git_ls_files(root, roots=tracked_roots)
    tracked_generated_files = [path for path in tracked_files if _match_any(path, tracked_patterns)]
    live_generated_files = _glob_repo(root, live_patterns)

    recommendations: List[str] = []
    if tracked_generated_files:
        recommendations.append(
            "Untrack generated runtime artifacts or move them into ignored runtime/report directories."
        )
    if live_generated_files:
        recommendations.append(
            "Keep runtime writers confined to ignored paths and preserve only curated reports under reports/."
        )

    return {
        "repo_root": str(root),
        "policy_path": policy["policy_path"],
        "tracked_generated_globs": tracked_patterns,
        "live_generated_globs": live_patterns,
        "tracked_generated_files": tracked_generated_files,
        "live_generated_files": live_generated_files,
        "tracked_violation_count": len(tracked_generated_files),
        "live_generated_count": len(live_generated_files),
        "recommendations": recommendations,
    }


def write_runtime_artifact_audit(
    repo_root: str | Path,
    output_path: str | Path,
    policy_path: str | Path | None = None,
) -> Dict[str, Any]:
    report = audit_runtime_artifacts(repo_root, policy_path=policy_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def render_runtime_artifact_audit_markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# Runtime Artifact Audit",
        "",
        f"- Repo root: `{report.get('repo_root')}`",
        f"- Policy path: `{report.get('policy_path')}`",
        f"- Tracked violation count: `{report.get('tracked_violation_count')}`",
        f"- Live generated count: `{report.get('live_generated_count')}`",
        "",
        "## Tracked Generated Files",
        "",
    ]
    tracked = report.get("tracked_generated_files") or []
    if tracked:
        for path in tracked:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Live Generated Files", ""])
    live = report.get("live_generated_files") or []
    if live:
        for path in live:
            lines.append(f"- `{path}`")
    else:
        lines.append("- None")
    lines.extend(["", "## Recommendations", ""])
    recommendations = report.get("recommendations") or []
    if recommendations:
        for item in recommendations:
            lines.append(f"1. {item}")
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_runtime_artifact_audit_markdown(
    report: Dict[str, Any],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render_runtime_artifact_audit_markdown(report), encoding="utf-8")
    return destination
