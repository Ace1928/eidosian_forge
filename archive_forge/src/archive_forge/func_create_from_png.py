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
@classmethod
def create_from_png(cls, source):
    """Decode a PNG file into a new image surface.

        :param source:
            A filename or
            a binary mode :term:`file object` with a ``read`` method.
            If you already have a byte string in memory,
            use :class:`io.BytesIO`.
        :returns: A new :class:`ImageSurface` instance.

        """
    if hasattr(source, 'read'):
        read_func = _make_read_func(source)
        pointer = cairo.cairo_image_surface_create_from_png_stream(read_func, ffi.NULL)
    else:
        pointer = cairo.cairo_image_surface_create_from_png(_encode_filename(source))
    self = object.__new__(cls)
    Surface.__init__(self, pointer)
    return self