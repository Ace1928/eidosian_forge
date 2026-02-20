#!/usr/bin/env python3
"""Generate a high-level index + catalog reference for the current user home."""

import argparse
import grp
import json
import os
import platform
import pwd
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from eidosian_core import eidosian

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.context_catalog import generate_catalog
from scripts.context_utils import CONTEXT_DIR, load_config, setup_logging

HOME = Path.home().resolve()
DEFAULT_INDEX_PATH = CONTEXT_DIR / "index.json"
DEFAULT_CATALOG_PATH = CONTEXT_DIR / "catalog.json"


@eidosian()
def parse_os_release():
    data = {}
    release_path = Path("/etc/os-release")
    if not release_path.exists():
        return data
    for line in release_path.read_text().splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key] = value.strip().strip('"')
    return data


@eidosian()
def run_command(cmd):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""
    return result.stdout.strip()


@eidosian()
def gather_system_performance():
    """Captures CPU load and RAM usage."""
    perf = {"cpu": {}, "memory": {}}

    # CPU
    try:
        load = os.getloadavg()
        perf["cpu"]["load_average"] = {"1m": load[0], "5m": load[1], "15m": load[2]}
        perf["cpu"]["count"] = os.cpu_count()
    except (OSError, AttributeError):
        pass

    # Memory
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip().split()[0]
                mem_info[key] = int(value) * 1024  # Convert to bytes

        if "MemTotal" in mem_info:
            total = mem_info["MemTotal"]
            free = mem_info.get("MemFree", 0) + mem_info.get("Buffers", 0) + mem_info.get("Cached", 0)
            used = total - free
            perf["memory"] = {
                "total_bytes": total,
                "used_bytes": used,
                "free_bytes": free,
                "used_percent": round((used / total) * 100, 1),
            }
    except (OSError, ValueError):
        pass

    return perf


@eidosian()
def git_info(path: Path):
    git_dir = path / ".git"
    try:
        if not git_dir.exists():
            return {}
    except PermissionError:
        return {}

    if shutil.which("git") is None:
        return {"error": "git command not found"}

    branch = run_command(["git", "-C", str(path), "rev-parse", "--abbrev-ref", "HEAD"])
    status = run_command(["git", "-C", str(path), "status", "-sb"])
    if not branch and not status:
        return {}
    return {"git_branch": branch, "git_status": status}


@eidosian()
def describe_entry(path: Path, manual_note, section_names, preview_limit=6):
    entry = {}
    relative_path = "." if path == HOME else str(path.relative_to(HOME))
    entry["name"] = path.name or relative_path
    entry["relative_path"] = relative_path
    entry["absolute_path"] = str(path.resolve())
    entry["type"] = (
        "directory"
        if path.is_dir() and not path.is_symlink()
        else "symlink" if path.is_symlink() else "file" if path.is_file() else "other"
    )
    entry["exists"] = path.exists()
    entry["section_names"] = sorted(section_names)
    entry["manual"] = manual_note or {}
    entry["statistics"] = {}
    entry["scan_errors"] = []

    try:
        stat_info = path.lstat()
    except OSError as exc:
        entry["scan_errors"].append(f"stat:{exc}")
        return entry

    entry["statistics"].update(
        {
            "size_bytes": stat_info.st_size,
            "last_modified": datetime.fromtimestamp(stat_info.st_mtime, tz=timezone.utc).isoformat(),
            "mode": oct(stat_info.st_mode & 0o777),
            "is_symlink": path.is_symlink(),
        }
    )

    try:
        owner = pwd.getpwuid(stat_info.st_uid).pw_name
    except KeyError:
        owner = stat_info.st_uid
    try:
        group = grp.getgrgid(stat_info.st_gid).gr_name
    except KeyError:
        group = stat_info.st_gid
    entry["statistics"].update({"owner": owner, "group": group})

    child_preview = []
    child_count = None
    preview_truncated = False
    if path.is_dir() and not path.is_symlink():
        try:
            children = sorted(path.iterdir(), key=lambda candidate: candidate.name.lower())
            child_count = len(children)
            preview_truncated = child_count > preview_limit
            for child in children[:preview_limit]:
                child_info = {
                    "name": child.name,
                    "relative_path": (str(child.relative_to(HOME)) if child != HOME else "."),
                    "type": (
                        "directory"
                        if child.is_dir() and not child.is_symlink()
                        else ("symlink" if child.is_symlink() else "file" if child.is_file() else "other")
                    ),
                }
                try:
                    child_info["size_bytes"] = child.stat().st_size
                except OSError:
                    child_info["size_bytes"] = None
                child_preview.append(child_info)
        except PermissionError as exc:
            entry["scan_errors"].append(f"iterdir:{exc}")
        except OSError as exc:
            entry["scan_errors"].append(f"iterdir:{exc}")
    entry["child_summary"] = {
        "count": child_count,
        "preview_limit": preview_limit,
        "preview": child_preview,
        "preview_truncated": preview_truncated,
    }

    git_meta = git_info(path)
    if git_meta:
        entry["git"] = git_meta

    pyvenv_marker = path / "pyvenv.cfg"
    try:
        if pyvenv_marker.exists():
            entry["pyvenv_marker"] = str(pyvenv_marker)
    except PermissionError as exc:
        entry["scan_errors"].append(f"pyvenv:{exc}")

    return entry


@eidosian()
def gather_directories(config, preview_limit=6):
    manual_notes = config.get("manual_notes", {})
    section_map = {}
    for section in config.get("sections", []):
        for entry_name in section.get("entries", []):
            section_map.setdefault(entry_name, []).append(section.get("name"))

    try:
        entries = [candidate for candidate in HOME.iterdir()]
        entries.sort(key=lambda candidate: candidate.name.lower())
    except PermissionError:
        entries = []

    data = []
    data.append(
        describe_entry(
            HOME,
            manual_notes.get("."),
            section_map.get(".", []),
            preview_limit=preview_limit,
        )
    )
    for entry in entries:
        manual_key = str(entry.relative_to(HOME))
        manual = manual_notes.get(manual_key) or manual_notes.get(entry.name)
        categories = section_map.get(manual_key) or section_map.get(entry.name, [])
        data.append(describe_entry(entry, manual, categories, preview_limit=preview_limit))
    return data


@eidosian()
def gather_storage_info():
    usage = shutil.disk_usage(HOME)
    filesystem = {}
    df_output = run_command(["df", "-P", str(HOME)])
    if df_output:
        lines = [line for line in df_output.splitlines() if line.strip()]
        if len(lines) > 1:
            headers = lines[0].split()
            values = lines[1].split()
            filesystem = dict(zip(headers, values))
    return {
        "path": str(HOME),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
        "used_percent": (round(usage.used / usage.total * 100, 1) if usage.total else None),
        "filesystem": filesystem,
    }


@eidosian()
def gather_metadata(config):
    uname = platform.uname()
    return {
        "root": str(HOME),
        "user": os.environ.get("USER", uname.node),
        "hostname": uname.node,
        "kernel": uname.release,
        "architecture": uname.machine,
        "shell": os.environ.get("SHELL"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "os_release": parse_os_release(),
        "path_entries": os.environ.get("PATH", "").split(os.pathsep),
        "config": config.get("index", {}),
    }


@eidosian()
def build_index(args):
    config = load_config()
    logger = setup_logging(config)
    logger.info("Starting context index generation (v2.0 - Performance Enhanced)")

    performance = gather_system_performance()
    structure = gather_directories(config, preview_limit=args.preview_limit)

    # Pass use_codex flag to generate_catalog
    catalog_payload = generate_catalog(config, logger, DEFAULT_CATALOG_PATH, use_codex=args.codex)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": str(Path(__file__).resolve()),
        "metadata": gather_metadata(config),
        "performance": performance,
        "storage": gather_storage_info(),
        "sections": config.get("sections", []),
        "notes": config.get("overview", {}).get("last_notes", []),
        "directories": structure,
        "virtual_environments": config.get("environments", []),
        "commands": config.get("commands", []),
        "supporting_files": config.get("supporting_files", {}),
        "manual_notes": config.get("manual_notes", {}),
        "catalog_reference": {
            "path": str(DEFAULT_CATALOG_PATH),
            "generated_at": catalog_payload.get("generated_at"),
            "profiling": catalog_payload.get("profiling", {}),
        },
    }
    output_path = Path(args.output).expanduser()
    msg = "Updating" if output_path.exists() else "Generating"
    logger.info("%s index at %s", msg, output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")
    return payload


@eidosian()
def parse_args():
    parser = argparse.ArgumentParser(description="Create a richly annotated context index for the home folder.")
    parser.add_argument(
        "--output",
        "-o",
        default=DEFAULT_INDEX_PATH,
        help="Path to write the generated JSON index.",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Skip the interactive notice when overwriting the index",
    )
    parser.add_argument(
        "--preview-limit",
        type=int,
        default=6,
        help="Number of child entries to capture per directory preview.",
    )
    parser.add_argument(
        "--codex",
        action="store_true",
        help="Use Codex (ChatMock) as the primary LLM provider instead of local Ollama.",
    )
    return parser.parse_args()


@eidosian()
def main():
    args = parse_args()
    build_index(args)


if __name__ == "__main__":
    main()
