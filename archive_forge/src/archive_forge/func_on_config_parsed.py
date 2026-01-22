from __future__ import annotations
import copy
import os
import secrets
import threading
from collections import OrderedDict
from typing import Any, Callable
from blinker import Signal
from streamlit import config_util, development, env_util, file_util, util
from streamlit.config_option import ConfigOption
from streamlit.errors import StreamlitAPIException
def on_config_parsed(func: Callable[[], None], force_connect=False, lock=False) -> Callable[[], bool]:
    """Wait for the config file to be parsed then call func.

    If the config file has already been parsed, just calls func immediately
    unless force_connect is set.

    Parameters
    ----------
    func : Callable[[], None]
        A function to run on config parse.

    force_connect : bool
        Wait until the next config file parse to run func, even if config files
        have already been parsed.

    lock : bool
        If set, grab _config_lock before running func.

    Returns
    -------
    Callable[[], bool]
        A function that the caller can use to deregister func.
    """
    receiver = lambda _: func_with_lock()

    def disconnect():
        return _on_config_parsed.disconnect(receiver)

    def func_with_lock():
        if lock:
            with _config_lock:
                func()
        else:
            func()
    if force_connect or not _config_options:
        _on_config_parsed.connect(receiver, weak=False)
    else:
        func_with_lock()
    return disconnect