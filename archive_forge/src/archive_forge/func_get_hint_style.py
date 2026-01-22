from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_hint_style(self):
    """Return the :ref:`HINT_STYLE` string for the font options object."""
    return cairo.cairo_font_options_get_hint_style(self._pointer)