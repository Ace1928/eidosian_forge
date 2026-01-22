from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def stroke(self):
    """A drawing operator that strokes the current path
        according to the current line width, line join, line cap,
        and dash settings.
        After :meth:`stroke`,
        the current path will be cleared from the cairo context.
        See :meth:`set_line_width`, :meth:`set_line_join`,
        :meth:`set_line_cap`, :meth:`set_dash`, and :meth:`stroke_preserve`.

        Note: Degenerate segments and sub-paths are treated specially
        and provide a useful result.
        These can result in two different situations:

        1. Zero-length "on" segments set in :meth:`set_dash`.
           If the cap style is :obj:`ROUND <LINE_CAP_ROUND>`
           or :obj:`SQUARE <LINE_CAP_SQUARE>`
           then these segments will be drawn
           as circular dots or squares respectively.
           In the case of :obj:`SQUARE <LINE_CAP_SQUARE>`,
           the orientation of the squares is determined
           by the direction of the underlying path.
        2. A sub-path created by :meth:`move_to` followed
           by either a :meth:`close_path`
           or one or more calls to :meth:`line_to`
           to the same coordinate as the :meth:`move_to`.
           If the cap style is :obj:`ROUND <LINE_CAP_ROUND>`
           then these sub-paths will be drawn as circular dots.
           Note that in the case of :obj:`SQUARE <LINE_CAP_SQUARE>`
           a degenerate sub-path will not be drawn at all,
           (since the correct orientation is indeterminate).

        In no case will a cap style of :obj:`BUTT <LINE_CAP_BUTT>`
        cause anything to be drawn
        in the case of either degenerate segments or sub-paths.

        """
    cairo.cairo_stroke(self._pointer)
    self._check_status()