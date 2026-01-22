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
def get_content(self):
    """Returns the :ref:`CONTENT` string of this surface,
        which indicates whether the surface contains color
        and/or alpha information.

        """
    return cairo.cairo_surface_get_content(self._pointer)