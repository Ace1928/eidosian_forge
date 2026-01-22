from __future__ import annotations
import collections
import os
import sys
import types
from pathlib import Path
from typing import Callable, Final
from streamlit import config, file_util
from streamlit.folder_black_list import FolderBlackList
from streamlit.logger import get_logger
from streamlit.source_util import get_pages
from streamlit.watcher.path_watcher import (
def on_file_changed(self, filepath):
    if filepath not in self._watched_modules:
        _LOGGER.error('Received event for non-watched file: %s', filepath)
        return
    for wm in self._watched_modules.values():
        if wm.module_name is not None and wm.module_name in sys.modules:
            del sys.modules[wm.module_name]
    for cb in self._on_file_changed:
        cb(filepath)