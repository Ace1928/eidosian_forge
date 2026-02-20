#!/usr/bin/env python3
import json
import os
from collections import defaultdict
from pathlib import Path

# Configuration
EXCLUDE_DIRS = {
    "archive_forge",
    "doc_forge",
    ".git",
    ".pytest_cache",
    "__pycache__",
    "eidosian_venv",
    ".vscode",
    "node_modules",
    ".eidos_chrome_profile",
    ".github",
    "Backups",
    "audit_data",
    "knowledge_cache",
    "logs",
    ".graphrag_index",
    "graphrag_workspace",
    "data",
}

# Whitelist for documentation
SOURCE_EXTENSIONS = {".py", ".md", ".js", ".ts", ".sh", ".c", ".cpp", ".h", ".hpp", ".txt", ".yaml", ".yml"}

ROOT_DIR = Path.cwd()
INDEX_FILE = ROOT_DIR / "doc_forge" / "file_index.json"


def should_document(path: Path) -> bool:
    """Check if the given path should be documented."""
    try:
        rel_path = path.relative_to(ROOT_DIR)
        for part in rel_path.parts:
            if part in EXCLUDE_DIRS:
                return False
    except ValueError:
        return False

    if path.suffix.lower() not in SOURCE_EXTENSIONS:
        return False

    try:
        if path.is_file() and path.stat().st_size > 500 * 1024:
            return False
    except OSError:
        return False

    return True


def scan_files():
    """Recursively scan for files and group them by directory."""
    dir_groups = defaultdict(list)
    print(f"DEBUG: Starting os.walk in {ROOT_DIR}")
    for root, dirs, files in os.walk(ROOT_DIR):
        root_path = Path(root)

        # Skip excluded directories in-place
        original_dirs = list(dirs)
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        # if len(original_dirs) != len(dirs):
        #    print(f"DEBUG: Skipping dirs in {root_path}: {[d for d in original_dirs if d not in dirs]}")

        rel_root = root_path.relative_to(ROOT_DIR)

        for file in files:
            file_path = root_path / file
            if should_document(file_path):
                dir_groups[str(rel_root)].append(file)
            # else:
            #    print(f"DEBUG: Skipping file {file_path}")

    return dir_groups


def main():
    print(f"Scanning {ROOT_DIR} for source and documentation files...")
    dir_groups = scan_files()

    total_files = sum(len(files) for files in dir_groups.values())
    print(f"Found {total_files} files across {len(dir_groups)} directories.")

    # Save the index to a JSON file
    with open(INDEX_FILE, "w") as f:
        json.dump(dir_groups, f, indent=2)
    print(f"File index saved to {INDEX_FILE}")

    # Display the first 10 directories and their files
    sorted_dirs = sorted(dir_groups.keys())
    for d in sorted_dirs[:10]:
        files = dir_groups[d]
        if files:
            print(f" - {d or '.'} ({len(files)} files)")
            for f in files[:3]:
                print(f"   - {f}")
            if len(files) > 3:
                print(f"   - ... and {len(files) - 3} more.")


if __name__ == "__main__":
    main()
