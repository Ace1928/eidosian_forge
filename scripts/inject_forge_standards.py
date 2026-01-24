from eidosian_core import eidosian
#!/usr/bin/env python3
"""
Injects Eidosian Forge standards (AGENTS.md, ruff config, pytest config) into a target directory.
Ensures consistency across the Eidosian ecosystem.
"""

import argparse
import sys
import shutil
from pathlib import Path

# Standard configurations
RUFF_CONFIG = """# Eidosian Forge - Standard Ruff Configuration
line-length = 120
target-version = "py312"

[lint]
select = ["E", "F", "I", "W"]
ignore = []
"""

PYTEST_CONFIG = """# Eidosian Forge - Standard Pytest Configuration
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v
"""

@eidosian()
def get_agents_md_source():
    # Try to find the reference AGENTS.md
    # Assuming this script is in scripts/ and AGENTS.md is in scripts/AGENTS.md
    script_dir = Path(__file__).resolve().parent
    candidates = [
        script_dir / "AGENTS.md",
        script_dir.parent / "AGENTS.md",
        Path.home() / "AGENTS.md"
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

@eidosian()
def inject_standards(target_dir: Path, force: bool = False):
    if not target_dir.exists():
        print(f"Error: Target directory {target_dir} does not exist.")
        sys.exit(1)

    print(f"Injecting standards into: {target_dir}")

    # 1. Inject AGENTS.md
    agents_source = get_agents_md_source()
    if agents_source:
        target_agents = target_dir / "AGENTS.md"
        if not target_agents.exists() or force:
            try:
                shutil.copy2(agents_source, target_agents)
                print(f"  [+] Copied AGENTS.md from {agents_source}")
            except Exception as e:
                print(f"  [!] Failed to copy AGENTS.md: {e}")
        else:
            print(f"  [.] AGENTS.md already exists (use --force to overwrite)")
    else:
        print("  [!] Could not locate source AGENTS.md to copy.")

    # 2. Inject ruff.toml
    target_ruff = target_dir / "ruff.toml"
    if not target_ruff.exists() or force:
        try:
            target_ruff.write_text(RUFF_CONFIG, encoding="utf-8")
            print(f"  [+] Created ruff.toml")
        except Exception as e:
            print(f"  [!] Failed to create ruff.toml: {e}")
    else:
        print(f"  [.] ruff.toml already exists (use --force to overwrite)")

    # 3. Inject pytest.ini
    target_pytest = target_dir / "pytest.ini"
    if not target_pytest.exists() or force:
        try:
            target_pytest.write_text(PYTEST_CONFIG, encoding="utf-8")
            print(f"  [+] Created pytest.ini")
        except Exception as e:
            print(f"  [!] Failed to create pytest.ini: {e}")
    else:
        print(f"  [.] pytest.ini already exists (use --force to overwrite)")

@eidosian()
def main():
    parser = argparse.ArgumentParser(description="Inject Eidosian Forge standards into a project.")
    parser.add_argument("target", nargs="?", default=".", help="Target directory (default: current)")
    parser.add_argument("--force", "-f", action="store_true", help="Overwrite existing files")
    
    args = parser.parse_args()
    target_path = Path(args.target).resolve()
    
    inject_standards(target_path, args.force)

if __name__ == "__main__":
    main()
