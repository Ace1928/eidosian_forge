#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from packaging.version import InvalidVersion, Version
except Exception:  # pragma: no cover - packaging may not be available everywhere
    InvalidVersion = Exception  # type: ignore[assignment]
    Version = None  # type: ignore[assignment]

SEVERITY_ORDER = {
    "critical": 4,
    "high": 3,
    "moderate": 2,
    "medium": 2,
    "low": 1,
    "unknown": 0,
}

PINNED_RE = re.compile(
    r"^(?P<prefix>\s*)"
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<extras>\[[^\]]+\])?"
    r"\s*==\s*"
    r"(?P<version>[^\s;#]+)"
    r"(?P<marker>\s*;[^#]*)?"
    r"(?P<comment>\s*#.*)?$"
)

GENERIC_REQ_RE = re.compile(
    r"^(?P<prefix>\s*)"
    r"(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?P<extras>\[[^\]]+\])?"
    r"(?P<rest>.*)$"
)


@dataclass(frozen=True)
class PatchTarget:
    package: str
    manifest_path: str
    target_version: str
    severity: str
    alert_numbers: tuple[int, ...]


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _normalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", str(name or "").strip().lower())


def _severity(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    if text == "moderate":
        return "medium"
    return text if text in SEVERITY_ORDER else "unknown"


def _version_tuple(value: str) -> tuple[Any, ...]:
    parts = re.split(r"[.\-+_]", value)
    out: list[Any] = []
    for part in parts:
        if part.isdigit():
            out.append(int(part))
        elif part:
            out.append(part)
    return tuple(out)


def _compare_versions(left: str, right: str) -> int:
    if left == right:
        return 0
    if Version is not None:
        try:
            lv = Version(left)
            rv = Version(right)
            if lv < rv:
                return -1
            if lv > rv:
                return 1
            return 0
        except InvalidVersion:
            pass
    lt = _version_tuple(left)
    rt = _version_tuple(right)
    if lt < rt:
        return -1
    if lt > rt:
        return 1
    return 0


def _load_alerts(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def collect_patch_targets(
    *,
    alerts: list[dict[str, Any]],
    allowed_states: set[str],
    allowed_severities: set[str],
    allowed_ecosystems: set[str],
    manifest_allowlist: set[str] | None = None,
) -> dict[str, dict[str, PatchTarget]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {}

    for row in alerts:
        state = str(row.get("state") or "").strip().lower()
        if state not in allowed_states:
            continue

        dep = row.get("dependency")
        if not isinstance(dep, dict):
            continue
        pkg = dep.get("package") if isinstance(dep.get("package"), dict) else {}
        ecosystem = str(pkg.get("ecosystem") or "").strip().lower()
        if ecosystem not in allowed_ecosystems:
            continue
        manifest_path = str(dep.get("manifest_path") or "").strip()
        if not manifest_path:
            continue
        if manifest_allowlist is not None and manifest_path not in manifest_allowlist:
            continue

        package_name = _normalize_name(str(pkg.get("name") or ""))
        if not package_name:
            continue

        vuln = row.get("security_vulnerability")
        if not isinstance(vuln, dict):
            continue
        severity = _severity(vuln.get("severity"))
        if severity not in allowed_severities:
            continue
        patch = vuln.get("first_patched_version") if isinstance(vuln.get("first_patched_version"), dict) else {}
        target = str(patch.get("identifier") or "").strip()
        if not target:
            continue
        number = int(row.get("number") or 0)

        manifest_map = grouped.setdefault(manifest_path, {})
        existing = manifest_map.get(package_name)
        if existing is None:
            manifest_map[package_name] = {
                "target_version": target,
                "severity": severity,
                "numbers": {number},
            }
            continue

        if _compare_versions(existing["target_version"], target) < 0:
            existing["target_version"] = target
        if SEVERITY_ORDER.get(severity, 0) > SEVERITY_ORDER.get(existing["severity"], 0):
            existing["severity"] = severity
        existing["numbers"].add(number)

    resolved: dict[str, dict[str, PatchTarget]] = {}
    for manifest, package_map in grouped.items():
        resolved[manifest] = {}
        for package, payload in package_map.items():
            resolved[manifest][package] = PatchTarget(
                package=package,
                manifest_path=manifest,
                target_version=str(payload["target_version"]),
                severity=str(payload["severity"]),
                alert_numbers=tuple(sorted(int(x) for x in payload["numbers"] if int(x) > 0)),
            )
    return resolved


def _render_updated_line(
    *,
    prefix: str,
    name: str,
    extras: str,
    target_version: str,
    marker: str,
    comment: str,
    newline: str,
) -> str:
    marker_part = marker or ""
    comment_part = comment or ""
    return f"{prefix}{name}{extras}=={target_version}{marker_part}{comment_part}{newline}"


def patch_requirements_file(
    path: Path,
    targets: dict[str, PatchTarget],
) -> tuple[str, dict[str, Any]]:
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)

    updated_lines: list[str] = []
    changed = False
    touched_packages: set[str] = set()
    unpinned_packages: set[str] = set()

    updates: list[dict[str, Any]] = []
    already_safe: list[dict[str, Any]] = []

    for index, line in enumerate(lines, start=1):
        newline = "\n" if line.endswith("\n") else ""
        body = line[:-1] if newline else line
        parsed = PINNED_RE.match(body)
        if parsed:
            name = parsed.group("name")
            norm = _normalize_name(name)
            if norm in targets:
                target = targets[norm]
                current_version = parsed.group("version")
                touched_packages.add(norm)
                if _compare_versions(current_version, target.target_version) >= 0:
                    already_safe.append(
                        {
                            "line": index,
                            "package": norm,
                            "current_version": current_version,
                            "target_version": target.target_version,
                        }
                    )
                    updated_lines.append(line)
                    continue
                updated = _render_updated_line(
                    prefix=parsed.group("prefix") or "",
                    name=name,
                    extras=parsed.group("extras") or "",
                    target_version=target.target_version,
                    marker=parsed.group("marker") or "",
                    comment=parsed.group("comment") or "",
                    newline=newline,
                )
                updated_lines.append(updated)
                changed = True
                updates.append(
                    {
                        "line": index,
                        "package": norm,
                        "from": current_version,
                        "to": target.target_version,
                        "severity": target.severity,
                        "alert_numbers": list(target.alert_numbers),
                    }
                )
                continue

        generic = GENERIC_REQ_RE.match(body)
        if generic:
            name = generic.group("name")
            norm = _normalize_name(name)
            if norm in targets:
                rest = (generic.group("rest") or "").lstrip()
                if rest.startswith("=="):
                    pass
                elif rest.startswith(("@", "!=", ">=", "<=", "~=", ">", "<", ";")) or not rest:
                    unpinned_packages.add(norm)

        updated_lines.append(line)

    unresolved_missing: list[str] = []
    unresolved_unpinned: list[str] = []
    for package in sorted(targets.keys()):
        if package in touched_packages:
            continue
        if package in unpinned_packages:
            unresolved_unpinned.append(package)
        else:
            unresolved_missing.append(package)

    patched = "".join(updated_lines)
    summary = {
        "path": path.as_posix(),
        "target_package_count": len(targets),
        "updated_count": len(updates),
        "already_safe_count": len(already_safe),
        "unresolved_missing_count": len(unresolved_missing),
        "unresolved_unpinned_count": len(unresolved_unpinned),
        "updates": updates,
        "already_safe": already_safe,
        "unresolved_missing": unresolved_missing,
        "unresolved_unpinned": unresolved_unpinned,
    }
    return patched, summary


def apply_patch_targets(
    *,
    repo_root: Path,
    patch_targets: dict[str, dict[str, PatchTarget]],
    write: bool,
    backup_root: Path | None,
) -> dict[str, Any]:
    report_manifests: list[dict[str, Any]] = []
    changed_files: list[str] = []
    backup_dir: Path | None = None

    if write and backup_root is not None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_dir = backup_root.expanduser().resolve() / stamp
        backup_dir.mkdir(parents=True, exist_ok=True)

    for manifest_path, targets in sorted(patch_targets.items()):
        full_path = (repo_root / manifest_path).resolve()
        if not full_path.exists():
            report_manifests.append(
                {
                    "path": manifest_path,
                    "error": "manifest_not_found",
                    "target_package_count": len(targets),
                    "updated_count": 0,
                    "already_safe_count": 0,
                    "unresolved_missing_count": len(targets),
                    "unresolved_unpinned_count": 0,
                    "updates": [],
                    "already_safe": [],
                    "unresolved_missing": sorted(targets.keys()),
                    "unresolved_unpinned": [],
                }
            )
            continue

        patched, summary = patch_requirements_file(full_path, targets)
        summary["path"] = manifest_path
        changed = any(x for x in summary["updates"])
        if write and changed:
            if backup_dir is not None:
                backup_path = backup_dir / manifest_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(full_path, backup_path)
            full_path.write_text(patched, encoding="utf-8")
            changed_files.append(manifest_path)

        summary["changed"] = bool(changed)
        report_manifests.append(summary)

    totals = {
        "manifest_count": len(report_manifests),
        "changed_manifest_count": sum(1 for row in report_manifests if row.get("changed")),
        "updated_requirements_count": sum(int(row.get("updated_count") or 0) for row in report_manifests),
        "already_safe_count": sum(int(row.get("already_safe_count") or 0) for row in report_manifests),
        "unresolved_missing_count": sum(int(row.get("unresolved_missing_count") or 0) for row in report_manifests),
        "unresolved_unpinned_count": sum(int(row.get("unresolved_unpinned_count") or 0) for row in report_manifests),
    }

    return {
        "generated_at": _now_iso(),
        "mode": "write" if write else "dry_run",
        "backup_dir": backup_dir.as_posix() if backup_dir is not None else "",
        "totals": totals,
        "changed_files": changed_files,
        "manifests": report_manifests,
    }


def render_markdown(report: dict[str, Any]) -> str:
    totals = report.get("totals") if isinstance(report.get("totals"), dict) else {}
    lines: list[str] = []
    lines.append("# Dependabot Requirement Auto-Patch Report")
    lines.append("")
    lines.append(f"- generated_at: `{report.get('generated_at')}`")
    lines.append(f"- mode: `{report.get('mode')}`")
    if report.get("backup_dir"):
        lines.append(f"- backup_dir: `{report.get('backup_dir')}`")
    lines.append(f"- manifest_count: `{totals.get('manifest_count', 0)}`")
    lines.append(f"- changed_manifest_count: `{totals.get('changed_manifest_count', 0)}`")
    lines.append(f"- updated_requirements_count: `{totals.get('updated_requirements_count', 0)}`")
    lines.append(f"- unresolved_missing_count: `{totals.get('unresolved_missing_count', 0)}`")
    lines.append(f"- unresolved_unpinned_count: `{totals.get('unresolved_unpinned_count', 0)}`")
    lines.append("")
    lines.append("## Manifest Summary")
    lines.append("")
    lines.append("| Manifest | Updated | Missing | Unpinned | Changed |")
    lines.append("|---|---:|---:|---:|---|")
    for row in report.get("manifests") or []:
        lines.append(
            "| `{path}` | `{u}` | `{m}` | `{up}` | `{changed}` |".format(
                path=row.get("path"),
                u=row.get("updated_count", 0),
                m=row.get("unresolved_missing_count", 0),
                up=row.get("unresolved_unpinned_count", 0),
                changed=row.get("changed", False),
            )
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch pinned pip requirements files using Dependabot first_patched_version metadata."
    )
    parser.add_argument(
        "--alerts-json",
        required=True,
        help="Path to raw dependabot alerts JSON (gh api output).",
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Repository root containing manifest paths (default: cwd).",
    )
    parser.add_argument(
        "--severity",
        default="critical,high",
        help="Comma-separated severities to include.",
    )
    parser.add_argument(
        "--state",
        default="open",
        help="Comma-separated states to include (default: open).",
    )
    parser.add_argument(
        "--ecosystem",
        default="pip",
        help="Comma-separated ecosystems to include (default: pip).",
    )
    parser.add_argument(
        "--manifest",
        action="append",
        default=[],
        help="Manifest path filter (repeat flag to target specific files).",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Apply changes to files; default is dry-run.",
    )
    parser.add_argument(
        "--backup-root",
        default="Backups/security_dependency_patches",
        help="Backup root for write mode.",
    )
    parser.add_argument("--json-out", default="", help="Optional output report JSON path.")
    parser.add_argument("--md-out", default="", help="Optional output report markdown path.")
    parser.add_argument(
        "--fail-on-unresolved",
        action="store_true",
        help="Exit non-zero if unresolved missing or unpinned packages remain.",
    )
    args = parser.parse_args()

    alerts_path = Path(args.alerts_json).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    alerts = _load_alerts(alerts_path)

    allowed_states = {str(x).strip().lower() for x in str(args.state).split(",") if str(x).strip()}
    if not allowed_states:
        allowed_states = {"open"}
    allowed_severities = {
        _severity(x)
        for x in str(args.severity).split(",")
        if str(x).strip()
    }
    allowed_severities.discard("unknown")
    if not allowed_severities:
        allowed_severities = {"critical", "high"}
    allowed_ecosystems = {
        str(x).strip().lower()
        for x in str(args.ecosystem).split(",")
        if str(x).strip()
    }
    if not allowed_ecosystems:
        allowed_ecosystems = {"pip"}
    manifest_allowlist = {str(x).strip() for x in args.manifest if str(x).strip()} or None

    targets = collect_patch_targets(
        alerts=alerts,
        allowed_states=allowed_states,
        allowed_severities=allowed_severities,
        allowed_ecosystems=allowed_ecosystems,
        manifest_allowlist=manifest_allowlist,
    )
    report = apply_patch_targets(
        repo_root=repo_root,
        patch_targets=targets,
        write=bool(args.write),
        backup_root=Path(args.backup_root) if args.backup_root else None,
    )
    report["filters"] = {
        "states": sorted(allowed_states),
        "severities": sorted(allowed_severities),
        "ecosystems": sorted(allowed_ecosystems),
        "manifest_allowlist": sorted(manifest_allowlist) if manifest_allowlist else [],
    }
    report["targets"] = {
        "manifest_count": len(targets),
        "package_count": sum(len(x) for x in targets.values()),
    }

    print(
        "dependabot_autopatch_requirements: "
        f"mode={report.get('mode')} "
        f"manifests={report.get('totals', {}).get('manifest_count', 0)} "
        f"changed={report.get('totals', {}).get('changed_manifest_count', 0)} "
        f"updated={report.get('totals', {}).get('updated_requirements_count', 0)} "
        f"unresolved={report.get('totals', {}).get('unresolved_missing_count', 0)}"
        f"/{report.get('totals', {}).get('unresolved_unpinned_count', 0)}"
    )

    if args.json_out:
        out = Path(args.json_out).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote json: {out}")

    if args.md_out:
        md = Path(args.md_out).expanduser().resolve()
        md.parent.mkdir(parents=True, exist_ok=True)
        md.write_text(render_markdown(report), encoding="utf-8")
        print(f"wrote md: {md}")

    unresolved = int(report.get("totals", {}).get("unresolved_missing_count", 0)) + int(
        report.get("totals", {}).get("unresolved_unpinned_count", 0)
    )
    if args.fail_on_unresolved and unresolved > 0:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
