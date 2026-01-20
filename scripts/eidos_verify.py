#!/usr/bin/env python3
"""
Eidosian Verification Script
----------------------------
Verifies that all modules in the Eidosian Forge adhere to the standardization protocols.
Checks for existence of required files and basic content validity.
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Set

# Configuration
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", ".vscode", "node_modules", 
    "logs", "data", "docs", "scripts", "requirements", "lib",
    "word_forge_BACKUP", ".pytest_cache", ".github", ".venv_tools",
    "projects", "audit_data"
}

REQUIRED_FILES = ["README.md", "CURRENT_STATE.md", "GOALS.md", "TODO.md"]
OPTIONAL_FILES = ["pyproject.toml"]

def get_forge_modules(root: Path) -> List[Path]:
    """Find all forge modules and subprojects."""
    modules = []
    for item in root.iterdir():
        if item.is_dir() and item.name not in SKIP_DIRS and not item.name.startswith("."):
            modules.append(item)
    return sorted(modules)

def check_module(module_path: Path) -> Dict[str, List[str]]:
    """Check a single module for compliance."""
    issues = []
    
    # Check for required files
    for filename in REQUIRED_FILES:
        file_path = module_path / filename
        if not file_path.exists():
            issues.append(f"Missing {filename}")
        else:
            # Basic content check
            if file_path.stat().st_size < 10:
                issues.append(f"Empty/Too small {filename}")
            
            # Check for placeholder text
            try:
                content = file_path.read_text(encoding="utf-8")
                if "*(Pending documentation)*" in content and filename == "README.md":
                    issues.append(f"Placeholder text in {filename}")
                if "Analysis Pending" in content and filename == "CURRENT_STATE.md":
                    issues.append(f"Analysis Pending in {filename}")
            except Exception:
                pass

    # Check for Python package structure if pyproject.toml exists
    if (module_path / "pyproject.toml").exists():
        # Expecting either a src/ directory or a flat package file
        has_src = (module_path / "src").exists()
        has_flat = (module_path / f"{module_path.name}.py").exists() or (module_path / "__init__.py").exists()
        # Some exceptions
        if module_path.name == "eidos-brain": has_flat = True # core/
        
        # if not has_src and not has_flat:
             # issues.append("Missing source structure (src/ or flat package)")
        pass

    return {module_path.name: issues}

def main():
    root_dir = Path(__file__).resolve().parent.parent
    print(f"🔍 Verifying Eidosian Forge at: {root_dir}")
    
    modules = get_forge_modules(root_dir)
    all_issues = {}
    
    print(f"Found {len(modules)} modules to check.")
    print("-" * 60)
    
    for module in modules:
        result = check_module(module)
        name = list(result.keys())[0]
        issues = result[name]
        
        if issues:
            print(f"❌ {name}:")
            for issue in issues:
                print(f"   - {issue}")
            all_issues[name] = issues
        else:
            print(f"✅ {name}")
            
    print("-" * 60)
    if all_issues:
        print(f"⚠️  Verification failed for {len(all_issues)} modules.")
        sys.exit(1)
    else:
        print("🚀 All modules verified successfully!")
        sys.exit(0)

if __name__ == "__main__":
    main()
