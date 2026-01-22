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
def migrate_static_custom(src: str, dst: str) -> bool:
    """Migrate non-empty custom.js,css from src to dst

    src, dst are 'custom' directories containing custom.{js,css}
    """
    log = get_logger()
    migrated = False
    custom_js = Path(src, 'custom.js')
    custom_css = Path(src, 'custom.css')
    custom_js_empty = True
    if Path(custom_js).is_file():
        with Path.open(custom_js, encoding='utf-8') as f:
            js = f.read().strip()
            for line in js.splitlines():
                if not (line.isspace() or line.strip().startswith(('/*', '*', '//'))):
                    custom_js_empty = False
                    break
    custom_css_empty = True
    if Path(custom_css).is_file():
        with Path.open(custom_css, encoding='utf-8') as f:
            css = f.read().strip()
            custom_css_empty = css.startswith('/*') and css.endswith('*/')
    if custom_js_empty:
        log.debug('Ignoring empty %s', custom_js)
    if custom_css_empty:
        log.debug('Ignoring empty %s', custom_css)
    if custom_js_empty and custom_css_empty:
        return False
    ensure_dir_exists(dst)
    if not custom_js_empty or not custom_css_empty:
        ensure_dir_exists(dst)
    if not custom_js_empty and migrate_file(custom_js, Path(dst, 'custom.js')):
        migrated = True
    if not custom_css_empty and migrate_file(custom_css, Path(dst, 'custom.css')):
        migrated = True
    return migrated