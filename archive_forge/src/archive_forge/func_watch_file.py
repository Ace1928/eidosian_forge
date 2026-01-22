from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def watch_file(path: str, on_file_changed: Callable[[str], None], watcher_type: str | None=None) -> bool:
    return _watch_path(path, on_file_changed, watcher_type)