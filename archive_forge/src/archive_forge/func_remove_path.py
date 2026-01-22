import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def remove_path(path: Path):
    """Remove path (file or directory) observing opt_dry_run."""
    if not path.exists():
        return
    if not opt_quiet:
        print(f'Removing {path.name}...')
    if opt_dry_run:
        return
    _remove_path_recursion(path)