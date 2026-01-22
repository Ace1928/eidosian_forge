import sys
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
from . import opt_dry_run, opt_quiet, QTPATHS_CMD, PROJECT_FILE_SUFFIX
def requires_rebuild(sources: List[Path], artifact: Path) -> bool:
    """Returns whether artifact needs to be rebuilt depending on sources"""
    if not artifact.is_file():
        return True
    artifact_mod_time = artifact.stat().st_mtime
    for source in sources:
        if source.stat().st_mtime > artifact_mod_time:
            return True
    return False