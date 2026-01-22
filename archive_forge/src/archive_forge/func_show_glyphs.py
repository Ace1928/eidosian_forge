from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def show_glyphs(self, glyphs):
    """A drawing operator that generates the shape from a list of glyphs,
        rendered according to the current
        font :meth:`face <set_font_face>`,
        font :meth:`size <set_font_size>`
        (font :meth:`matrix <set_font_matrix>`),
        and font :meth:`options <set_font_options>`.

        :param glyphs:
            The glyphs to show.
            See :meth:`show_text_glyphs` for the data structure.

        """
    glyphs = ffi.new('cairo_glyph_t[]', glyphs)
    cairo.cairo_show_glyphs(self._pointer, glyphs, len(glyphs))
    self._check_status()