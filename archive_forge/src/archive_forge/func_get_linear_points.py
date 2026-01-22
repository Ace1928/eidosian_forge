from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
from .surfaces import Surface
def get_linear_points(self):
    """Return this linear gradient’s endpoints.

        :returns: A ``(x0, y0, x1, y1)`` tuple of floats.

        """
    points = ffi.new('double[4]')
    _check_status(cairo.cairo_pattern_get_linear_points(self._pointer, points + 0, points + 1, points + 2, points + 3))
    return tuple(points)