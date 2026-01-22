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
def migrate_file(src: str | Path, dst: str | Path, substitutions: Any=None) -> bool:
    """Migrate a single file from src to dst

    substitutions is an optional dict of {regex: replacement} for performing replacements on the file.
    """
    log = get_logger()
    if Path(dst).exists():
        log.debug('%s already exists', dst)
        return False
    log.info('Copying %s -> %s', src, dst)
    ensure_dir_exists(Path(dst).parent)
    shutil.copy(src, dst)
    if substitutions:
        with Path.open(Path(dst), encoding='utf-8') as f:
            text = f.read()
        for pat, replacement in substitutions.items():
            text = pat.sub(replacement, text)
        with Path.open(Path(dst), 'w', encoding='utf-8') as f:
            f.write(text)
    return True