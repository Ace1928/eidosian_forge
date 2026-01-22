from . import _check_status, cairo, ffi
def transform_point(self, x, y):
    """Transforms the point ``(x, y)`` by this matrix.

        :param x: X position.
        :param y: Y position.
        :type x: float
        :type y: float
        :returns: A ``(new_x, new_y)`` tuple of floats.

        """
    xy = ffi.new('double[2]', [x, y])
    cairo.cairo_matrix_transform_point(self._pointer, xy + 0, xy + 1)
    return tuple(xy)