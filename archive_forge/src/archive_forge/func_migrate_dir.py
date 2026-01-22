from __future__ import annotations
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from traitlets.config.loader import JSONFileConfigLoader, PyFileConfigLoader
from traitlets.log import get_logger
from .application import JupyterApp
from .paths import jupyter_config_dir, jupyter_data_dir
from .utils import ensure_dir_exists
def migrate_dir(src: str, dst: str) -> bool:
    """Migrate a directory from src to dst"""
    log = get_logger()
    if not os.listdir(src):
        log.debug('No files in %s', src)
        return False
    if Path(dst).exists():
        if os.listdir(dst):
            log.debug('%s already exists', dst)
            return False
        Path(dst).rmdir()
    log.info('Copying %s -> %s', src, dst)
    ensure_dir_exists(Path(dst).parent)
    shutil.copytree(src, dst, symlinks=True)
    return True