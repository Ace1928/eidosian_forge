from __future__ import annotations
import os
import sys
from typing import Any
import click
import streamlit.runtime.caching as caching
import streamlit.runtime.legacy_caching as legacy_caching
import streamlit.web.bootstrap as bootstrap
from streamlit import config as _config
from streamlit.config_option import ConfigOption
from streamlit.runtime.credentials import Credentials, check_credentials
from streamlit.web.cache_storage_manager_config import (
def configurator_options(func):
    """Decorator that adds config param keys to click dynamically."""
    for _, value in reversed(_config._config_options_template.items()):
        parsed_parameter = _convert_config_option_to_click_option(value)
        if value.sensitive:
            click_option_kwargs = {'expose_value': False, 'hidden': True, 'is_eager': True, 'callback': _make_sensitive_option_callback(value)}
        else:
            click_option_kwargs = {'show_envvar': True, 'envvar': parsed_parameter['envvar']}
        config_option = click.option(parsed_parameter['option'], parsed_parameter['param'], help=parsed_parameter['description'], type=parsed_parameter['type'], **click_option_kwargs)
        func = config_option(func)
    return func