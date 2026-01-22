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
@config.command('show')
@configurator_options
def config_show(**kwargs):
    """Show all of Streamlit's config settings."""
    bootstrap.load_config_options(flag_options=kwargs)
    _config.show_config()