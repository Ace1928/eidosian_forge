from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_scaled_font(self, scaled_font):
    """Replaces the current font face, font matrix, and font options
        with those of ``scaled_font``.
        Except for some translation, the current CTM of the context
        should be the same as that of the ``scaled_font``,
        which can be accessed using :meth:`ScaledFont.get_ctm`.

        :param scaled_font: A :class:`ScaledFont` object.

        """
    cairo.cairo_set_scaled_font(self._pointer, scaled_font._pointer)
    self._check_status()