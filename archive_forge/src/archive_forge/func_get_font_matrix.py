from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def get_font_matrix(self):
    """Copies the scaled font’s font matrix.

        :returns: A new :class:`Matrix` object.

        """
    matrix = Matrix()
    cairo.cairo_scaled_font_get_font_matrix(self._pointer, matrix._pointer)
    self._check_status()
    return matrix