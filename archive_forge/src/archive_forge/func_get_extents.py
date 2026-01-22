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
def get_extents(self):
    """Return the extents of the recording-surface.

        :returns:
            A ``(x, y, width, height)`` tuple of floats,
            or :obj:`None` if the surface is unbounded.

        *New in cairo 1.12*

        """
    extents = ffi.new('cairo_rectangle_t *')
    if cairo.cairo_recording_surface_get_extents(self._pointer, extents):
        return (extents.x, extents.y, extents.width, extents.height)