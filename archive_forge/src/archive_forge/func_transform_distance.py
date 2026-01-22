from . import _check_status, cairo, ffi
def transform_distance(self, dx, dy):
    """Transforms the distance vector ``(dx, dy)`` by this matrix.
        This is similar to :meth:`transform_point`
        except that the translation components of the transformation
        are ignored.
        The calculation of the returned vector is as follows::

            dx2 = dx1 * xx + dy1 * xy
            dy2 = dx1 * yx + dy1 * yy

        Affine transformations are position invariant,
        so the same vector always transforms to the same vector.
        If ``(x1, y1)`` transforms to ``(x2, y2)``
        then ``(x1 + dx1, y1 + dy1)`` will transform
        to ``(x1 + dx2, y1 + dy2)`` for all values of ``x1`` and ``x2``.

        :param dx: X component of a distance vector.
        :param dy: Y component of a distance vector.
        :type dx: float
        :type dy: float
        :returns: A ``(new_dx, new_dy)`` tuple of floats.

        """
    xy = ffi.new('double[2]', [dx, dy])
    cairo.cairo_matrix_transform_distance(self._pointer, xy + 0, xy + 1)
    return tuple(xy)