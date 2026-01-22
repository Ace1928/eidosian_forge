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
def format_stride_for_width(format, width):
    """
        This method provides a stride value (byte offset between rows)
        that will respect all alignment requirements
        of the accelerated image-rendering code within cairo.
        Typical usage will be of the form::

            from cairocffi import ImageSurface
            stride = ImageSurface.format_stride_for_width(format, width)
            data = bytearray(stride * height)
            surface = ImageSurface(format, width, height, data, stride)

        :param format: A :ref:`FORMAT` string.
        :param width: The desired width of the surface, in pixels.
        :type format: str
        :type width: int
        :returns:
            The appropriate stride to use given the desired format and width,
            or -1 if either the format is invalid or the width too large.

        """
    return cairo.cairo_format_stride_for_width(format, width)