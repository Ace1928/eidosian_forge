from __future__ import annotations
from contextlib import contextmanager
from typing import Any
from streamlit import config
def build_mock_config_get_option(overrides_dict):
    orig_get_option = config.get_option

    def mock_config_get_option(name):
        if name in overrides_dict:
            return overrides_dict[name]
        return orig_get_option(name)
    return mock_config_get_option