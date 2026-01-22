from __future__ import annotations
import random
from importlib.metadata import version as _version
from typing import TYPE_CHECKING, Final
import streamlit.logger as logger
def should_show_new_version_notice() -> bool:
    """True if streamlit should show a 'new version!' notice to the user.

    We need to make a network call to PyPI to determine the latest streamlit
    version. Since we don't want to do this every time streamlit is run,
    we'll only perform the check ~5% of the time.

    If we do make the request to PyPI and there's any sort of error,
    we log it and return False.

    Returns
    -------
    bool
        True if we should tell the user that their streamlit is out of date.

    """
    if random.random() >= CHECK_PYPI_PROBABILITY:
        _LOGGER.debug('Skipping PyPI version check')
        return False
    try:
        installed_version = _get_installed_streamlit_version()
        latest_version = _get_latest_streamlit_version(timeout=1)
    except Exception as ex:
        _LOGGER.debug('Failed PyPI version check.', exc_info=ex)
        return False
    return latest_version > installed_version