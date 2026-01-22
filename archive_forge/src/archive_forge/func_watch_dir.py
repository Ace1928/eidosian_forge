from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def watch_dir(path: str, on_dir_changed: Callable[[str], None], watcher_type: str | None=None, *, glob_pattern: str | None=None, allow_nonexistent: bool=False) -> bool:
    return _watch_path(path, on_dir_changed, watcher_type, glob_pattern=glob_pattern, allow_nonexistent=allow_nonexistent)