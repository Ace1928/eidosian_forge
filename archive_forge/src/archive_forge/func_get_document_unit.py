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
def get_document_unit(self):
    """Get the unit of the SVG surface.

        If the surface passed as an argument is not a SVG surface, the function
        sets the error status to ``STATUS_SURFACE_TYPE_MISMATCH`` and
        returns :data:`SVG_UNIT_USER`.

        :return: The SVG unit of the SVG surface.

        *New in cairo 1.16.*

        *New in cairocffi 0.9.*

        """
    unit = cairo.cairo_svg_surface_get_document_unit(self._pointer)
    self._check_status()
    return unit