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
def mark_dirty_rectangle(self, x, y, width, height):
    """
        Like :meth:`mark_dirty`,
        but drawing has been done only to the specified rectangle,
        so that cairo can retain cached contents
        for other parts of the surface.

        Any cached clip set on the surface will be reset by this method,
        to make sure that future cairo calls have the clip set
        that they expect.

        :param x: X coordinate of dirty rectangle.
        :param y: Y coordinate of dirty rectangle.
        :param width: Width of dirty rectangle.
        :param height: Height of dirty rectangle.
        :type x: float
        :type y: float
        :type width: float
        :type height: float

        """
    cairo.cairo_surface_mark_dirty_rectangle(self._pointer, x, y, width, height)
    self._check_status()