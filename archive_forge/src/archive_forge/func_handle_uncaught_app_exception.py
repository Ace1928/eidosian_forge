from __future__ import annotations
from typing import Final
import streamlit as st
from streamlit import config
from streamlit.errors import UncaughtAppException
from streamlit.logger import get_logger
def handle_uncaught_app_exception(ex: BaseException) -> None:
    """Handle an exception that originated from a user app.

    By default, we show exceptions directly in the browser. However,
    if the user has disabled client error details, we display a generic
    warning in the frontend instead.
    """
    error_logged = False
    if config.get_option('logger.enableRich'):
        try:
            _print_rich_exception(ex)
            error_logged = True
        except Exception:
            error_logged = False
    if config.get_option('client.showErrorDetails'):
        if not error_logged:
            _LOGGER.warning('Uncaught app exception', exc_info=ex)
        st.exception(ex)
    else:
        if not error_logged:
            _LOGGER.error('Uncaught app exception', exc_info=ex)
        st.exception(UncaughtAppException(ex))