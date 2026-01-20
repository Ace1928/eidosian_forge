#!/usr/bin/env python3
"""
Eidosian Standardization Script
-------------------------------
Automates the creation of standard documentation and structure across all
Eidosian Forge modules.

Usage:
    python scripts/eidos_standardize.py
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Any

# Add root to sys.path to import global_info
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

try:
    import global_info
except ImportError:
    print("❌ Critical: Could not import global_info.py from root.")
    sys.exit(1)

# Configuration
SKIP_DIRS = {
    "__pycache__", ".git", ".venv", ".vscode", "node_modules", 
    "logs", "data", "docs", "scripts", "requirements", "lib",
    "word_forge_BACKUP"
}

STANDARD_FILES = ["README.md", "CURRENT_STATE.md", "GOALS.md", "TODO.md"]

def get_forge_modules(root: Path) -> List[Path]:
    """Find all forge modules and subprojects."""
    modules = []
    for item in root.iterdir():
        if item.is_dir() and item.name not in SKIP_DIRS:
            # Criteria: ends with _forge OR is in specific list
            if item.name.endswith("_forge") or item.name in ["eidos-brain", "eidos_mcp", "graphrag"]:
                modules.append(item)
    return sorted(modules)

def generate_readme(module_name: str, path: Path) -> str:
    """Generate a standardized README content."""
    title = module_name.replace("_", " ").title()
    if module_name == "eidos-brain":
        title = "Eidos Brain"
    
    return f"""# {title}

**Part of the Eidosian Forge**

## 📋 Overview
The `{module_name}` module is a specialized component of the Eidosian Intelligence System.
*(Auto-generated: Please add specific description here)*

## 📂 Structure
- `src/` (Recommended): Source code
- `tests/`: Unit and integration tests

## 🛠️ Usage
*(Pending documentation)*

## 🧪 Testing
Run tests from the project root:
```bash
pytest {module_name}/
```
"""

def generate_current_state(module_name: str, path: Path) -> str:
    """Generate CURRENT_STATE.md content."""
    # Simple analysis
    py_files = list(path.rglob("*.py"))
    file_count = len(py_files)
    
    return f"""# Current State: {module_name}

**Date**: {global_info.date.today().strftime('%Y-%m-%d')}
**Status**: Analysis Pending

## 📊 Metrics
- **Python Files**: {file_count}
- **Test Coverage**: Unknown

## 🏗️ Architecture
*(Describe the current architectural state)*

## 🐛 Known Issues
- Documentation needs update.
- Type coverage verification needed.
"""

def generate_goals(module_name: str) -> str:
    """Generate GOALS.md content."""
    return f"""# Goals: {module_name}

## 🎯 Immediate Goals
- [ ] Complete `README.md` documentation.
- [ ] Verify 100% type safety (`mypy`).
- [ ] Ensure 85%+ test coverage.

## 🔭 Long-term Vision
- Integrate fully with `eidos-brain`.
- Optimize for performance and scalability.
"""

def generate_todo(module_name: str) -> str:
    """Generate TODO.md content."""
    return f"""# TODO: {module_name}

## 🚨 High Priority
- [ ] **Audit**: specific files in this module.
- [ ] **Docs**: Fill in `README.md` Overview.
- [ ] **Style**: Run `black` and `isort`.

## 🟡 Medium Priority
- [ ] Add docstrings to all public functions.
- [ ] Create unit tests for core logic.

## 🟢 Low Priority
- [ ] Refactor long functions.
"""

def process_module(module_path: Path):
    """Check and create standard files for a module."""
    print(f"🔍 Processing {module_path.name}...")
    
    generators = {
        "README.md": generate_readme,
        "CURRENT_STATE.md": generate_current_state,
        "GOALS.md": generate_goals,
        "TODO.md": generate_todo
    }
    
    for filename, generator in generators.items():
        file_path = module_path / filename
        if not file_path.exists():
            print(f"  ➕ Creating {filename}")
            if filename in ["CURRENT_STATE.md", "README.md"]:
                content = generator(module_path.name, module_path)
            else:
                content = generator(module_path.name)
            
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                print(f"  ❌ Error writing {filename}: {e}")
        else:
            # print(f"  ✅ {filename} exists")
            pass

def main():
    print(f"🚀 Eidosian Standardization Tool v{global_info.get_version()}")
    print(f"📂 Root: {root_dir}")
    
    modules = get_forge_modules(root_dir)
    print(f"found {len(modules)} modules.")
    
    for module in modules:
        process_module(module)
        
    print("\n✅ Standardization check complete.")

if __name__ == "__main__":
    main()
