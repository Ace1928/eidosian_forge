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
def dsc_begin_page_setup(self):
    """Indicate that subsequent calls to :meth:`dsc_comment` should
        direct comments to the PageSetup section of the PostScript output.

        This method is only needed for the first page of a surface.
        It must be called after any call to :meth:`dsc_begin_setup`
        and before any drawing is performed to the surface.

        See :meth:`dsc_comment` for more details.

        """
    cairo.cairo_ps_surface_dsc_begin_page_setup(self._pointer)
    self._check_status()