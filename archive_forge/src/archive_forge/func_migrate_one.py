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
def migrate_one(src: str, dst: str) -> bool:
    """Migrate one item

    dispatches to migrate_dir/_file
    """
    log = get_logger()
    if Path(src).is_file():
        return migrate_file(src, dst)
    if Path(src).is_dir():
        return migrate_dir(src, dst)
    log.debug('Nothing to migrate for %s', src)
    return False