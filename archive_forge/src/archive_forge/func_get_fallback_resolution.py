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
def get_fallback_resolution(self):
    """Returns the previous fallback resolution
        set by :meth:`set_fallback_resolution`,
        or default fallback resolution if never set.

        :returns: ``(x_pixels_per_inch, y_pixels_per_inch)``

        """
    ppi = ffi.new('double[2]')
    cairo.cairo_surface_get_fallback_resolution(self._pointer, ppi + 0, ppi + 1)
    return tuple(ppi)