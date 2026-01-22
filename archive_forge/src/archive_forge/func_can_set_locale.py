from __future__ import annotations
from contextlib import contextmanager
import locale
import platform
import re
import subprocess
from typing import TYPE_CHECKING
from pandas._config.config import options
def can_set_locale(lc: str, lc_var: int=locale.LC_ALL) -> bool:
    """
    Check to see if we can set a locale, and subsequently get the locale,
    without raising an Exception.

    Parameters
    ----------
    lc : str
        The locale to attempt to set.
    lc_var : int, default `locale.LC_ALL`
        The category of the locale being set.

    Returns
    -------
    bool
        Whether the passed locale can be set
    """
    try:
        with set_locale(lc, lc_var=lc_var):
            pass
    except (ValueError, locale.Error):
        return False
    else:
        return True