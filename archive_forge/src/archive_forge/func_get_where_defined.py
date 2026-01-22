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
def get_where_defined(key: str) -> str:
    """Indicate where (e.g. in which file) this option was defined.

    Parameters
    ----------
    key : str
        The config option key of the form "section.optionName"

    """
    with _config_lock:
        config_options = get_config_options()
        if key not in config_options:
            raise RuntimeError('Config key "%s" not defined.' % key)
        return config_options[key].where_defined