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
def get_device_scale(self):
    """Returns the previous device offset set by :meth:`set_device_scale`.

        *New in cairo 1.14.*

        *New in cairocffi 0.9.*

        """
    size = ffi.new('double[2]')
    cairo.cairo_surface_get_device_scale(self._pointer, size + 0, size + 1)
    return tuple(size)