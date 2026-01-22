import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def package_dir() -> Path:
    """Return the PySide6 root."""
    return Path(__file__).resolve().parents[2]