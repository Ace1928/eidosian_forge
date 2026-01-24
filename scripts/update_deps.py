from eidosian_core import eidosian
#!/usr/bin/env python3
"""
update_deps.py - Update pyproject.toml dependencies to match installed versions.

This script reads the current virtual environment's installed packages and updates
the dependency constraints in pyproject.toml files to reflect the installed versions.

Usage:
    python update_deps.py --root <path/to/forge_root> [--strategy <caret|gte|pinned>] [--dry-run]

Strategies:
    caret:  Sets version to "^X.Y.Z" (e.g., ^1.2.3)
    gte:    Sets version to ">=X.Y.Z" (e.g., >=1.2.3)
    pinned: Sets version to "==X.Y.Z" (e.g., ==1.2.3)
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

@eidosian()
def parse_args():
    parser = argparse.ArgumentParser(description="Update dependencies from venv")
    parser.add_argument("--root", required=True, type=Path, help="Root directory of the forge")
    parser.add_argument("--strategy", choices=["caret", "gte", "pinned"], default="gte",
                        help="Versioning strategy")
    parser.add_argument("--dry-run", action="store_true", help="Don't write changes")
    return parser.parse_args()

@eidosian()
def get_installed_packages():
    """Returns a dict of installed packages {name: version} using pip list."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "list", "--format=json"],
            capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)
        return {pkg["name"].lower(): pkg["version"] for pkg in data}
    except Exception as e:
        print(f"Error getting installed packages: {e}")
        sys.exit(1)

@eidosian()
def update_file(file_path: Path, installed: dict, strategy: str, dry_run: bool):
    content = file_path.read_text(encoding="utf-8")
    original_content = content
    
    # regex for: key = "value"
    # capturing key (group 1), quote (group 2), value (group 3), end quote (group 4)
    pattern = re.compile(r'^(\s*([a-zA-Z0-9_-]+)\s*=\s*)([""])(.*?)([""])', re.MULTILINE)

    @eidosian()
    def replacement(match):
        prefix = match.group(1)
        pkg_name = match.group(2).lower()
        quote = match.group(3)
        current_ver = match.group(4)
        end_quote = match.group(5)

        # Skip python constraint
        if pkg_name == "python":
            return match.group(0)

        # Check if package is installed
        if pkg_name in installed:
            installed_ver = installed[pkg_name]
            
            # Determine new constraint string
            if strategy == "caret":
                new_ver = f"^{installed_ver}"
            elif strategy == "pinned":
                new_ver = f"=={installed_ver}"
            else: # gte
                new_ver = f">={installed_ver}"
                
            # Avoid updating if already set (heuristic check could be better)
            if current_ver == new_ver:
                return match.group(0)
                
            print(f"  [{file_path.parent.name}] {pkg_name}: {current_ver} -> {new_ver}")
            return f"{prefix}{quote}{new_ver}{end_quote}"
        
        return match.group(0)

    new_content = pattern.sub(replacement, content)

    if new_content != original_content:
        if not dry_run:
            file_path.write_text(new_content, encoding="utf-8")
            print(f"Updated {file_path}")
    
@eidosian()
def main():
    args = parse_args()
    installed = get_installed_packages()
    print(f"Loaded {len(installed)} installed packages.")
    
    # Find all pyproject.toml files
    for toml_file in args.root.rglob("pyproject.toml"):
        # Skip hidden dirs and venv
        if ".venv" in str(toml_file) or "eidosian_venv" in str(toml_file):
            continue
            
        print(f"Scanning {toml_file}...")
        update_file(toml_file, installed, args.strategy, args.dry_run)

if __name__ == "__main__":
    main()
