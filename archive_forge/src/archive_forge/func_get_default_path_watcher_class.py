from __future__ import annotations
from typing import Callable, Type, Union
import streamlit.watcher
from streamlit import cli_util, config, env_util
from streamlit.watcher.polling_path_watcher import PollingPathWatcher
def get_default_path_watcher_class() -> PathWatcherType:
    """Return the class to use for path changes notifications, based on the
    server.fileWatcherType config option.
    """
    return get_path_watcher_class(config.get_option('server.fileWatcherType'))