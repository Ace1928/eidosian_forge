#!/usr/bin/env python3
"""
Version Update Utility - Dynamic Version Management 🔢

Updates version numbers across the codebase from a single source of truth.
Uses XML-style version tags for precise location and automated updates.

Usage: python tools/version_update.py X.Y.Z
"""

import re
import sys
import os
import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any, Set


# ╭──────────────────────────────────────────────────────╮
# │  🔍 Version Pattern Definitions                      │
# ╰──────────────────────────────────────────────────────╯

VERSION_TAG_START = "<!-- VERSION_START -->"
VERSION_TAG_END = "<!-- VERSION_END -->"

# Current date in ISO format
CURRENT_DATE = datetime.date.today().isoformat()
DATE_FORMAT = "%Y-%m-%d"  # ISO format: YYYY-MM-DD
FORMATTED_DATE = "2025-03-12"  # Today's date as specified

# Pattern configurations for various file types
FILE_PATTERNS = {
    # Python file version patterns
    ".py": {
        "direct": r'__version__\s*=\s*[\'"]([^\'"]+)[\'"](\s*#\s*FALLBACK_VERSION)',
        "comment": r'#\s*Version:\s*([0-9.]+)',
        "date_comment": r'#\s*Last updated:\s*([0-9-]+)',
    },
    # Toml file version patterns
    ".toml": {
        "direct": r'(version\s*=\s*")([^"]+)(".*?AUTO-MANAGED)',
        "date": r'(date\s*=\s*")([^"]+)(")',
    },
    # Markdown file version patterns
    ".md": {
        "badge": r'(\!\[Version\]\()https://img\.shields\.io/badge/version-([0-9.]+)',
        "date_badge": r'(\!\[Updated\]\()https://img\.shields\.io/badge/updated-([0-9-]+)',
        "header": r'(#+\s*Version:\s*)([0-9.]+)',
        "date": r'(Last updated:\s*)([0-9-]+)',
        "text": r'(Current version:\s*)([0-9.]+)',
    },
    # YAML file version patterns
    ".yml": {
        "version": r'(\s+version:\s*)([0-9.]+)',
        "date": r'(\s+date:\s*)([0-9-]+)',
    },
    ".yaml": {
        "version": r'(\s+version:\s*)([0-9.]+)',
        "date": r'(\s+date:\s*)([0-9-]+)',
    }
}


# ╭──────────────────────────────────────────────────────╮
# │  🔄 File Update Functions                            │
# ╰──────────────────────────────────────────────────────╯

def update_file_version(file_path: Path, new_version: str) -> bool:
    """Update version in a file based on its type.
    
    Args:
        file_path: Path to file
        new_version: New version string
        
    Returns:
        True if updated successfully, False otherwise
    """
    if not file_path.exists():
        print(f"Error: {file_path} not found")
        return False
    
    content = file_path.read_text(encoding="utf-8")
    suffix = file_path.suffix.lower()
    updated = False
    
    # Check for version tag blocks first
    if VERSION_TAG_START in content and VERSION_TAG_END in content:
        updated_content = update_version_tag_block(content, new_version)
        if updated_content != content:
            content = updated_content
            updated = True
    
    # Try file-specific patterns
    if suffix in FILE_PATTERNS:
        patterns = FILE_PATTERNS[suffix]
        
        for pattern_type, pattern in patterns.items():
            if "date" in pattern_type.lower():
                # Update date patterns with current date
                content = re.sub(pattern, lambda m: f'{m.group(1)}{FORMATTED_DATE}{m.group(3) if len(m.groups()) > 2 else ""}', content)
            elif pattern_type == "direct" or "version" in pattern_type.lower():
                # Update version patterns
                if len(re.findall(pattern, content)) > 0:  # Check if pattern exists in content
                    content = re.sub(pattern, lambda m: f'{m.group(1)}{new_version}{m.group(3) if len(m.groups()) > 2 else ""}', content)
                    updated = True
    
    if updated:
        file_path.write_text(content, encoding="utf-8")
        print(f"✓ Updated {file_path}")
        return True
    else:
        print(f"ℹ️ No version patterns found in {file_path}")
        return False


def update_version_tag_block(content: str, new_version: str) -> str:
    """Update version between version tag blocks.
    
    Args:
        content: File content
        new_version: New version string
        
    Returns:
        Updated content
    """
    # Find all blocks between version tags
    pattern = f"{VERSION_TAG_START}(.*?){VERSION_TAG_END}"
    
    def replace_version_in_block(match):
        block = match.group(1)
        
        # Replace version in common patterns within the block
        patterns = [
            (r'__version__\s*=\s*[\'"]([^\'"]+)[\'"]', f'__version__ = "{new_version}"'),
            (r'version\s*=\s*"([^"]+)"', f'version = "{new_version}"'),
            (r'version\s*=\s*\'([^\']+)\'', f"version = '{new_version}'"),
            (r'(Current version:)\s*([0-9.]+)', f'\\1 {new_version}'),
            (r'(Last updated:)\s*([0-9-]+)', f'\\1 {FORMATTED_DATE}'),
            # Update version in version badge
            (r'(\!\[Version\]\()https://img\.shields\.io/badge/version-([0-9.]+)', f'\\1https://img.shields.io/badge/version-{new_version}'),
        ]
        
        for pattern, replacement in patterns:
            block = re.sub(pattern, replacement, block)
        
        # Update dates
        date_pattern = r'(Last updated|Updated on|Date:)(\s*:?\s*)([0-9]{4}-[0-9]{2}-[0-9]{2})'
        block = re.sub(date_pattern, f'\\1\\2{FORMATTED_DATE}', block)
            
        return f"{VERSION_TAG_START}{block}{VERSION_TAG_END}"
    
    return re.sub(pattern, replace_version_in_block, content, flags=re.DOTALL)


def update_pyproject_version(file_path: Path, new_version: str) -> bool:
    """Update version in pyproject.toml file.
    
    Args:
        file_path: Path to pyproject.toml
        new_version: New version string
        
    Returns:
        True if updated successfully, False otherwise
    """
    return update_file_version(file_path, new_version)


# ╭──────────────────────────────────────────────────────╮
# │  🔎 File Discovery and Validation                    │
# ╰──────────────────────────────────────────────────────╯

def find_versionable_files(project_root: Path) -> Set[Path]:
    """Find files that might contain version information.
    
    Args:
        project_root: Project root directory
        
    Returns:
        Set of files to check for versioning
    """
    # Key project files that definitely need version updates
    key_files = [
        project_root / "pyproject.toml",
        project_root / "__init__.py",
        project_root / "refactor_forge/__init__.py",
        project_root / "README.md",
        project_root / "CHANGELOG.md",
    ]
    
    versioned_files = set()
    
    # Add existing key files
    for file in key_files:
        if file.exists():
            versioned_files.add(file)
    
    # Find all Python files that might have version info
    for py_file in project_root.rglob("*.py"):
        # Skip test files and tools except version_update.py
        if (not py_file.name.startswith("test_") and 
            not (py_file.name != "version_update.py" and "tools" in str(py_file))):
            with open(py_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Only add files with version patterns
                if re.search(r"__version__|# Version:|VERSION_TAG", content):
                    versioned_files.add(py_file)
    
    # Add additional documentation files
    for md_file in project_root.rglob("*.md"):
        # Skip certain files
        if md_file.name not in ["CONTRIBUTING.md", "CODE_OF_CONDUCT.md"]:
            versioned_files.add(md_file)
    
    # Add workflow files
    for workflow_file in (project_root / ".github" / "workflows").glob("*.yml"):
        if workflow_file.exists():
            versioned_files.add(workflow_file)
    
    return versioned_files


def validate_version(version: str) -> bool:
    """Validate that version string follows semantic versioning.
    
    Args:
        version: Version string to validate
        
    Returns:
        True if valid, False otherwise
    """
    # Simple semver pattern: X.Y.Z with optional prerelease/build metadata
    pattern = r"^\d+\.\d+\.\d+(-[0-9A-Za-z-]+)?(\+[0-9A-Za-z-]+)?$"
    return bool(re.match(pattern, version))


# ╭──────────────────────────────────────────────────────╮
# │  🚀 Main Function & Orchestration                    │
# ╰──────────────────────────────────────────────────────╯

def sync_version(new_version: str, project_root: Optional[Path] = None) -> Dict[str, bool]:
    """Synchronize version across all relevant project files.
    
    Args:
        new_version: New version string
        project_root: Project root directory (defaults to script's parent dir)
        
    Returns:
        Dictionary mapping file paths to success status
    """
    if project_root is None:
        # Default to two directories up from this script
        project_root = Path(__file__).parent.parent
    
    results = {}
    
    # Start with pyproject.toml as the primary source of truth
    pyproject_path = project_root / "pyproject.toml"
    if pyproject_path.exists():
        results[str(pyproject_path)] = update_pyproject_version(pyproject_path, new_version)
    
    # Find and update all versionable files
    versionable_files = find_versionable_files(project_root)
    for file_path in versionable_files:
        if file_path != pyproject_path:  # Skip if already processed
            results[str(file_path)] = update_file_version(file_path, new_version)
    
    return results


def main() -> int:
    """Main entry point for the script.
    
    Returns:
        Exit code (0 for success, 1 for errors)
    """
    if len(sys.argv) != 2:
        print("Usage: python version_update.py X.Y.Z")
        return 1
    
    new_version = sys.argv[1]
    
    if not validate_version(new_version):
        print(f"Error: '{new_version}' is not a valid semantic version (X.Y.Z)")
        return 1
    
    print(f"🔄 Updating version to {new_version} with date {FORMATTED_DATE}...")
    results = sync_version(new_version)
    
    # Report results
    success_count = sum(1 for result in results.values() if result)
    fail_count = len(results) - success_count
    
    print(f"\n✅ Updated {success_count} files successfully")
    if fail_count > 0:
        print(f"⚠️ {fail_count} files could not be updated. Check warnings above.")
    
    if success_count > 0:
        print(f"\n🚀 Version {new_version} (dated {FORMATTED_DATE}) is now set across the codebase!")
        return 0
    else:
        print("\n❌ No files were updated. Please check file paths and version patterns.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
