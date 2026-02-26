#!/usr/bin/env python3
"""Eidosian Doc Processor v3 Entrypoint.

Delegates to the modular doc_forge.scribe package.
"""

import sys
from pathlib import Path

# Ensure the package is in path
script_dir = Path(__file__).resolve().parent
forge_root = script_dir.parent.parent
src_dir = forge_root / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from doc_forge.scribe.service import main

if __name__ == "__main__":
    sys.exit(main())
