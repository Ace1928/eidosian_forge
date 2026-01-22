import ctypes
import io
import operator
import os
import sys
import weakref
from functools import reduce
from pathlib import Path
from tempfile import NamedTemporaryFile
from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontOptions, _encode_string
@staticmethod
def ps_level_to_string(level):
    """Return the string representation of the given :ref:`PS_LEVEL`.
        See :meth:`get_levels` for a way to get
        the list of valid level ids.

        """
    c_string = cairo.cairo_ps_level_to_string(level)
    if c_string == ffi.NULL:
        raise ValueError(level)
    return ffi.string(c_string).decode('ascii')