from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_antialias(self):
    """Return the :ref:`ANTIALIAS` string for the font options object."""
    return cairo.cairo_font_options_get_antialias(self._pointer)