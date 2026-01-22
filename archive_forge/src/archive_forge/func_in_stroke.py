from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def in_stroke(self, x, y):
    """Tests whether the given point is inside the area
        that would be affected by a :meth:`stroke` operation
        given the current path and stroking parameters.
        Surface dimensions and clipping are not taken into account.

        See :meth:`stroke`, :meth:`set_line_width`, :meth:`set_line_join`,
        :meth:`set_line_cap`, :meth:`set_dash`, and :meth:`stroke_preserve`.

        :param x: X coordinate of the point to test
        :param y: Y coordinate of the point to test
        :type x: float
        :type y: float
        :returns: A boolean.

        """
    return bool(cairo.cairo_in_stroke(self._pointer, x, y))