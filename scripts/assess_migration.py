from eidosian_core import eidosian
#!/usr/bin/env python3
"""
Migration Assessment Tool.

Analyzes a directory in /home/lloyd/Development and scores it for
integration into eidosian_forge.
"""

import sys
import os
from pathlib import Path

@eidosian()
def assess_directory(path_str):
    path = Path(path_str)
    if not path.exists():
        print(f"Error: {path} does not exist.")
        return

    score = 0
    reasons = []

    # Criteria 1: Git Repo
    if (path / ".git").exists():
        score += 2
        reasons.append("Is a git repository (+2)")
    
    # Criteria 2: Python Project
    if (path / "requirements.txt").exists() or (path / "pyproject.toml").exists() or (path / "setup.py").exists():
        score += 3
        reasons.append("Is a Python project (+3)")
    
    # Criteria 3: Documentation
    if (path / "README.md").exists():
        score += 1
        reasons.append("Has README (+1)")

    # Criteria 4: Structure
    if (path / "src").exists() or (path / "tests").exists():
        score += 2
        reasons.append("Has src/tests structure (+2)")

    print(f"Assessment for: {path.name}")
    print(f"Path: {path}")
    print(f"Score: {score}/8")
    print("Reasons:")
    for r in reasons:
        print(f"  - {r}")

    if score >= 5:
        print("RECOMMENDATION: High candidate for integration.")
    elif score >= 3:
        print("RECOMMENDATION: Potential candidate, requires manual review.")
    else:
        print("RECOMMENDATION: Low priority or unstructured data.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 assess_migration.py <path_to_directory>")
        sys.exit(1)
    
    assess_directory(sys.argv[1])
