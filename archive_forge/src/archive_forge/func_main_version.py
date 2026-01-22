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
@main.command('version')
def main_version():
    """Print Streamlit's version number."""
    import sys
    _get_command_line_as_string()
    assert len(sys.argv) == 2
    sys.argv[1] = '--version'
    main()