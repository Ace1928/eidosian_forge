#!/usr/bin/env python3
"""Interactive command-line overview for the home context index."""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from eidosian_core import eidosian

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "context" / "index.json"
CONTEXT_SCRIPT = PROJECT_ROOT / "scripts" / "context_index.py"


@eidosian()
def load_index(path: Path):
    try:
        with path.open() as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise SystemExit(
            f"Index file not found: {path}. Run python scripts/context_index.py first."
        )


@eidosian()
def refresh_index(index_path: Path):
    print("Refreshing context index...")
    result = subprocess.run(
        [sys.executable, CONTEXT_SCRIPT, "--force", "--output", str(index_path)]
    )
    if result.returncode != 0:
        raise SystemExit("Failed to refresh the context index.")


@eidosian()
def print_summary(data):
    meta = data.get("metadata", {})
    storage = data.get("storage", {})
    print("\nHome Context Overview")
    print("----------------------")
    print(f"Root: {meta.get('root')} ({meta.get('hostname')})")
    print(
        f"User: {meta.get('user')} | Shell: {meta.get('shell')} | Python: {meta.get('python_version')} @ {meta.get('python_executable')}"
    )
    fs = storage.get("filesystem", {})
    if fs:
        print(
            f"Filesystem: {fs.get('Filesystem', fs.get('filesystem'))} mounted on {fs.get('Mounted', fs.get('Mounted on', storage.get('path')))}"
        )
    print(
        f"Disk Usage: {storage.get('used_bytes')} used of {storage.get('total_bytes')} ({storage.get('used_percent')}%)"
    )
    sections = data.get("sections", [])
    if sections:
        names = ", ".join(section.get("name") for section in sections)
        print(f"Sections: {names}")
    print()
    print("Commands (type 'c' to list):")
    for entry in data.get("commands", [])[:5]:
        print(
            f"  {entry.get('name')}: {entry.get('cmd')} - {entry.get('description', '---')}"
        )
    print()


@eidosian()
def print_directory_list(directories):
    print("Directories: (enter number to view details)")
    for idx, entry in enumerate(directories):
        desc = entry.get("manual", {}).get(
            "description", "No manual description provided."
        )
        print(
            f"  [{idx:02d}] {entry.get('relative_path')} ({entry.get('type')}) - {desc}"
        )
    print()


@eidosian()
def print_storage(storage):
    fs = storage.get("filesystem", {})
    lines = [
        f"Path: {storage.get('path')}",
        f"Used: {storage.get('used_bytes')} / {storage.get('total_bytes')} bytes",
    ]
    if fs:
        lines.append(
            f"Filesystem details: {' | '.join(f'{k}={v}' for k, v in fs.items())}"
        )
    print("\n".join(lines))
    print()


@eidosian()
def print_commands(commands):
    print("Known commands:")
    for cmd in commands:
        desc = cmd.get("description", "(no description)")
        print(f"  {cmd.get('name')}: {cmd.get('cmd')} - {desc}")
    print()


@eidosian()
def show_entry_detail(entry):
    print()
    print(f"{entry.get('relative_path')} ({entry.get('type')})")
    print("  " + (entry.get("manual", {}).get("description") or "No manual note."))
    tags = entry.get("manual", {}).get("tags", [])
    if tags:
        print(f"  Tags: {', '.join(tags)}")
    stats = entry.get("statistics", {})
    print(
        f"  Size: {stats.get('size_bytes')} bytes | Modified: {stats.get('last_modified')}"
    )
    if entry.get("git"):
        print(
            f"  Git: {entry['git'].get('git_branch')} | {entry['git'].get('git_status')}"
        )
    if entry.get("pyvenv_marker"):
        print(f"  Pyvenv: {entry.get('pyvenv_marker')}")
    child_summary = entry.get("child_summary", {})
    count = child_summary.get("count")
    preview = child_summary.get("preview", [])
    print(f"  Immediate children: {count}")
    if preview:
        print("  Preview:")
        for child in preview:
            print(f"    - {child.get('name')} ({child.get('type')})")
    if entry.get("scan_errors"):
        print("  Scan errors:")
        for err in entry.get("scan_errors"):
            print(f"    * {err}")
    print()


@eidosian()
def interactive_loop(data, index_path: Path):
    directories = data.get("directories", [])
    commands = data.get("commands", [])
    print_directory_list(directories)
    help_text = "Commands: [number] show entry, c=commands, s=storage, r=refresh, l=list, q=quit"
    print(help_text)
    while True:
        try:
            user = input("context> ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            break
        if not user:
            continue
        if user in {"q", "quit"}:
            break
        if user in {"c", "commands"}:
            print_commands(commands)
            continue
        if user in {"s", "storage"}:
            print_storage(data.get("storage", {}))
            continue
        if user in {"l", "list"}:
            print_directory_list(directories)
            continue
        if user in {"r", "refresh"}:
            refresh_index(index_path)
            data = load_index(index_path)
            directories = data.get("directories", [])
            commands = data.get("commands", [])
            print_summary(data)
            print_directory_list(directories)
            continue
        if user.isdigit():
            idx = int(user)
            if 0 <= idx < len(directories):
                show_entry_detail(directories[idx])
                continue
            print(f"Invalid index: {idx}")
            continue
        print("Unknown command. " + help_text)


@eidosian()
def main():
    parser = argparse.ArgumentParser(
        description="Command-line UI for the home context index."
    )
    parser.add_argument(
        "--index",
        "-i",
        default=DEFAULT_INDEX_PATH,
        help="Path to the generated index JSON.",
    )
    parser.add_argument(
        "--refresh",
        "-r",
        action="store_true",
        help="Regenerate the index before launching the UI.",
    )
    args = parser.parse_args()

    index_path = Path(args.index).expanduser()
    if args.refresh:
        refresh_index(index_path)
    data = load_index(index_path)
    print_summary(data)
    interactive_loop(data, index_path)


if __name__ == "__main__":
    main()
