from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_font_face(self, font_face):
    """Replaces the current font face with ``font_face``.

        :param font_face:
            A :class:`FontFace` object,
            or :obj:`None` to restore the default font.

        """
    font_face = font_face._pointer if font_face is not None else ffi.NULL
    cairo.cairo_set_font_face(self._pointer, font_face)
    self._check_status()