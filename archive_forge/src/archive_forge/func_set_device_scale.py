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
def set_device_scale(self, x_scale, y_scale):
    """Sets a scale that is multiplied to the device coordinates determined
        by the CTM when drawing to surface.

        One common use for this is to render to very high resolution display
        devices at a scale factor, so that code that assumes 1 pixel will be a
        certain size will still work.  Setting a transformation via
        cairo_translate() isn't sufficient to do this, since functions like
        cairo_device_to_user() will expose the hidden scale.

        Note that the scale affects drawing to the surface as well as using the
        surface in a source pattern.

        :param x_scale: the scale in the X direction, in device units.
        :param y_scale: the scale in the Y direction, in device units.

        *New in cairo 1.14.*

        *New in cairocffi 0.9.*

        """
    cairo.cairo_surface_set_device_scale(self._pointer, x_scale, y_scale)
    self._check_status()