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
def restrict_to_level(self, level):
    """Restricts the generated PostScript file to ``level``.

        See :meth:`get_levels` for a list of available level values
        that can be used here.

        This method should only be called
        before any drawing operations have been performed on the given surface.
        The simplest way to do this is to call this method
        immediately after creating the surface.

        :param version: A :ref:`PS_LEVEL` string.

        """
    cairo.cairo_ps_surface_restrict_to_level(self._pointer, level)
    self._check_status()