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
def get_eps(self):
    """Check whether the PostScript surface will output
        Encapsulated PostScript.

        """
    return bool(cairo.cairo_ps_surface_get_eps(self._pointer))