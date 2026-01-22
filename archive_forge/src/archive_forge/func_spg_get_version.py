from __future__ import annotations
import warnings
from typing import TYPE_CHECKING
import numpy as np
def spg_get_version():
    """Get the X.Y.Z version of the detected spglib C library.

    .. versionadded:: 2.3.0
    :return: version string
    """
    _set_no_error()
    return _spglib.version_string()