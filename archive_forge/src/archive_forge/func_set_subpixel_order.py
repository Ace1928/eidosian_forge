from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def set_subpixel_order(self, subpixel_order):
    """Changes the :ref:`SUBPIXEL_ORDER` for the font options object.
         The subpixel order specifies the order of color elements
         within each pixel on the display device
         when rendering with an antialiasing mode of
         :obj:`SUBPIXEL <ANTIALIAS_SUBPIXEL>`.

        """
    cairo.cairo_font_options_set_subpixel_order(self._pointer, subpixel_order)
    self._check_status()