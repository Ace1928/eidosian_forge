from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
def check_readable(mode):
    """Check a mode string allows reading.

    Arguments:
        mode (str): A mode string, e.g. ``"rt"``

    Returns:
        bool: `True` if the mode allows reading.

    """
    return Mode(mode).reading