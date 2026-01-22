from __future__ import print_function, unicode_literals
import typing
import six
from ._typing import Text
def check_writable(mode):
    """Check a mode string allows writing.

    Arguments:
        mode (str): A mode string, e.g. ``"wt"``

    Returns:
        bool: `True` if the mode allows writing.

    """
    return Mode(mode).writing