from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_subpixel_order(self):
    """Return the :ref:`SUBPIXEL_ORDER` string
        for the font options object.

        """
    return cairo.cairo_font_options_get_subpixel_order(self._pointer)