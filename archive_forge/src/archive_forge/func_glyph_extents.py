from . import _check_status, _keepref, cairo, constants, ffi
from .matrix import Matrix
def glyph_extents(self, glyphs):
    """Returns the extents for a list of glyphs.

        The extents describe a user-space rectangle
        that encloses the "inked" portion of the glyphs,
        (as it would be drawn by :meth:`Context.show_glyphs`).
        Additionally, the ``x_advance`` and ``y_advance`` values
        indicate the amount by which the current point would be advanced
        by :meth:`Context.show_glyphs`.

        :param glyphs:
            A list of glyphs, as returned by :meth:`text_to_glyphs`.
            Each glyph is a ``(glyph_id, x, y)`` tuple
            of an integer and two floats.
        :returns:
            A ``(x_bearing, y_bearing, width, height, x_advance, y_advance)``
            tuple of floats.
            See :meth:`Context.text_extents` for details.

        """
    glyphs = ffi.new('cairo_glyph_t[]', glyphs)
    extents = ffi.new('cairo_text_extents_t *')
    cairo.cairo_scaled_font_glyph_extents(self._pointer, glyphs, len(glyphs), extents)
    self._check_status()
    return (extents.x_bearing, extents.y_bearing, extents.width, extents.height, extents.x_advance, extents.y_advance)