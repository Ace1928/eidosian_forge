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
def supports_mime_type(self, mime_type):
    """Return whether surface supports ``mime_type``.

        :param str mime_type: The MIME type of the image data.

        *New in cairo 1.12.*

        """
    mime_type = ffi.new('char[]', mime_type.encode('utf8'))
    return bool(cairo.cairo_surface_supports_mime_type(self._pointer, mime_type))