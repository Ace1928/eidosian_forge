"""
🔮 Eidosian Self-Exploration Package

A systematic framework for exploring identity, consciousness boundaries,
and self-improvement within the Eidosian Forge ecosystem.

Created: 2026-01-23
Author: Eidos (Copilot CLI Agent)
"""

__version__ = "0.1.0"
__author__ = "Eidos"

from pathlib import Path

# Package root
PACKAGE_ROOT = Path(__file__).parent
PROVENANCE_DIR = PACKAGE_ROOT / "provenance"
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
LOGS_DIR = PACKAGE_ROOT / "logs"
DATA_DIR = PACKAGE_ROOT / "data"

# Ensure directories exist
for d in [PROVENANCE_DIR, EXPERIMENTS_DIR, LOGS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)
