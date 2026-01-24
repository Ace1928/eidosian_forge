from eidosian_core import eidosian
#!/usr/bin/env python3
"""
relax_deps.py - Relax dependency constraints in pyproject.toml

This script updates dependency definitions in a pyproject.toml file to make them
less strict, which is useful when integrating multiple projects into a single
monorepo environment.

Usage:
    python relax_deps.py --file <path/to/pyproject.toml> --package <package_name> [--strategy <>=|*>]

Strategies:
    gte: Changes "^1.2.3" to ">=1.2.3" (default)
    star: Changes "^1.2.3" to "*"
    none: Removes version constraint entirely (same as *)
"""

import argparse
import re
import shutil
import sys
from pathlib import Path

@eidosian()
def parse_args():
    parser = argparse.ArgumentParser(description="Relax dependency constraints in pyproject.toml")
    parser.add_argument("--file", required=True, type=Path, help="Path to pyproject.toml")
    parser.add_argument("--package", required=True, help="Package name to relax")
    parser.add_argument("--strategy", choices=["gte", "star", "none"], default="gte", 
                        help="Relaxation strategy (gte='>=', star='*')")
    parser.add_argument("--backup", action="store_true", default=True, help="Create a backup file")
    return parser.parse_args()

@eidosian()
def relax_dependency(content: str, package: str, strategy: str) -> str:
    """
    Finds the dependency line and updates its constraint.
    Assumes format: package = "constraint" or package = { ... }
    """
    # Regex to find simple string dependencies: package = "^1.2.3"
    # Group 1: key
    # Group 2: quote
    # Group 3: version
    # Group 4: quote
    pattern_simple = re.compile(rf'^(\s*{re.escape(package)}\s*=\s*)(["\'])(.*?)(["\'])', re.MULTILINE)
    
    @eidosian()
    def replacement_simple(match):
        prefix = match.group(1)
        quote = match.group(2)
        version = match.group(3)
        end_quote = match.group(4)
        
        # Determine new version string
        if strategy == "star" or strategy == "none":
            new_version = "*"
        else: # gte
            # If it starts with ^ or ~, replace with >=
            if version.startswith("^") or version.startswith("~"):
                 new_version = ">=" + version[1:]
            # If it's exact ==, replace with >=
            elif version.startswith("=="):
                 new_version = ">=" + version[2:]
            # If it's already >= or *, leave it (or just clean it up)
            elif version.startswith(">=") or version == "*":
                 new_version = version
            else:
                 # Assume it's an exact version if no operator, so make it >=
                 # But poetry usually implies exact if no operator? No, poetry usually requires caret.
                 # Let's assume ^ if it looks like a version number
                 if re.match(r'^\d', version):
                     new_version = ">=" + version
                 else:
                     new_version = version # Don't touch unknown formats
        
        print(f"Relaxing {package}: {version} -> {new_version}")
        return f"{prefix}{quote}{new_version}{end_quote}"

    new_content = pattern_simple.sub(replacement_simple, content)
    
    if new_content == content:
        print(f"Warning: Could not find or update dependency '{package}' in standard format.")
        # TODO: Handle table format if necessary: package = { version = "..." }
    
    return new_content

@eidosian()
def main():
    args = parse_args()
    
    if not args.file.exists():
        print(f"Error: File not found: {args.file}")
        sys.exit(1)
        
    content = args.file.read_text(encoding="utf-8")
    
    if args.backup:
        backup_path = args.file.with_suffix(".toml.bak")
        shutil.copy2(args.file, backup_path)
        print(f"Backup created at {backup_path}")
        
    new_content = relax_dependency(content, args.package, args.strategy)
    
    if new_content != content:
        args.file.write_text(new_content, encoding="utf-8")
        print(f"Successfully updated {args.file}")
    else:
        print("No changes made.")

if __name__ == "__main__":
    main()
