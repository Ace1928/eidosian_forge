from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_tolerance(self, tolerance):
    """Sets the tolerance used when converting paths into trapezoids.
        Curved segments of the path will be subdivided
        until the maximum deviation between the original path
        and the polygonal approximation is less than tolerance.
        The default value is 0.1.
        A larger value will give better performance,
        a smaller value, better appearance.
        (Reducing the value from the default value of 0.1
        is unlikely to improve appearance significantly.)
        The accuracy of paths within Cairo is limited
        by the precision of its internal arithmetic,
        and the prescribed tolerance is restricted
        to the smallest representable internal value.

        :type tolerance: float
        :param tolerance: The tolerance, in device units (typically pixels)

        """
    cairo.cairo_set_tolerance(self._pointer, tolerance)
    self._check_status()