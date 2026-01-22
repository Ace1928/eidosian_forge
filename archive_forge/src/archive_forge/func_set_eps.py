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
def set_eps(self, eps):
    """
        If ``eps`` is True,
        the PostScript surface will output Encapsulated PostScript.

        This method should only be called
        before any drawing operations have been performed on the current page.
        The simplest way to do this is to call this method
        immediately after creating the surface.
        An Encapsulated PostScript file should never contain
        more than one page.

        """
    cairo.cairo_ps_surface_set_eps(self._pointer, bool(eps))
    self._check_status()