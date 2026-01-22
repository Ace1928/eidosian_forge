from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def paint_with_alpha(self, alpha):
    """A drawing operator that paints the current source everywhere
        within the current clip region
        using a mask of constant alpha value alpha.
        The effect is similar to :meth:`paint`,
        but the drawing is faded out using the ``alpha`` value.

        :type alpha: float
        :param alpha: Alpha value, between 0 (transparent) and 1 (opaque).

        """
    cairo.cairo_paint_with_alpha(self._pointer, alpha)
    self._check_status()