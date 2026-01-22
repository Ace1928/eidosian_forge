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
@main.command('run')
@configurator_options
@click.argument('target', required=True, envvar='STREAMLIT_RUN_TARGET')
@click.argument('args', nargs=-1)
def main_run(target: str, args=None, **kwargs):
    """Run a Python script, piping stderr to Streamlit.

    The script can be local or it can be an url. In the latter case, Streamlit
    will download the script to a temporary file and runs this file.

    """
    from streamlit import url_util
    bootstrap.load_config_options(flag_options=kwargs)
    _, extension = os.path.splitext(target)
    if extension[1:] not in ACCEPTED_FILE_EXTENSIONS:
        if extension[1:] == '':
            raise click.BadArgumentUsage('Streamlit requires raw Python (.py) files, but the provided file has no extension.\nFor more information, please see https://docs.streamlit.io')
        else:
            raise click.BadArgumentUsage(f'Streamlit requires raw Python (.py) files, not {extension}.\nFor more information, please see https://docs.streamlit.io')
    if url_util.is_url(target):
        from streamlit.temporary_directory import TemporaryDirectory
        with TemporaryDirectory() as temp_dir:
            from urllib.parse import urlparse
            path = urlparse(target).path
            main_script_path = os.path.join(temp_dir, path.strip('/').rsplit('/', 1)[-1])
            target = url_util.process_gitblob_url(target)
            _download_remote(main_script_path, target)
            _main_run(main_script_path, args, flag_options=kwargs)
    else:
        if not os.path.exists(target):
            raise click.BadParameter(f'File does not exist: {target}')
        _main_run(target, args, flag_options=kwargs)