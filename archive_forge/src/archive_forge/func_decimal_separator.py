from __future__ import annotations
import gettext as gettext_module
import os.path
from threading import local
def decimal_separator() -> str:
    """Return the decimal separator for a locale, default to dot.

    Returns:
         str: Decimal separator.
    """
    try:
        sep = _DECIMAL_SEPARATOR[_CURRENT.locale]
    except (AttributeError, KeyError):
        sep = '.'
    return sep