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
def get_options_for_section(section: str) -> dict[str, Any]:
    """Get all of the config options for the given section.

    Run `streamlit config show` in the terminal to see all available options.

    Parameters
    ----------
    section : str
        The name of the config section to fetch options for.

    Returns
    -------
    dict[str, Any]
        A dict mapping the names of the options in the given section (without
        the section name as a prefix) to their values.
    """
    with _config_lock:
        config_options = get_config_options()
        options_for_section = {}
        for option in config_options.values():
            if option.section == section:
                options_for_section[option.name] = option.value
        return options_for_section