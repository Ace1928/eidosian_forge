from __future__ import annotations
import os
from typing import Final, NoReturn
import streamlit as st
from streamlit import source_util
from streamlit.deprecation_util import make_deprecated_name_warning
from streamlit.errors import NoSessionContext, StreamlitAPIException
from streamlit.file_util import get_main_script_directory, normalize_path_join
from streamlit.logger import get_logger
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.scriptrunner import RerunData, get_script_run_ctx
@gather_metrics('experimental_rerun')
def experimental_rerun() -> NoReturn:
    """Rerun the script immediately.

    When ``st.experimental_rerun()`` is called, the script is halted - no
    more statements will be run, and the script will be queued to re-run
    from the top.
    """
    msg = make_deprecated_name_warning('experimental_rerun', 'rerun', '2024-04-01')
    _LOGGER.warning(msg)
    rerun()