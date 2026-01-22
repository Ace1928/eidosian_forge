from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_line_join(self, line_join):
    """Set the current :ref:`LINE_JOIN` within the cairo context.
        As with the other stroke parameters,
        the current line cap style is examined by
        :meth:`stroke` and :meth:`stroke_extents`,
        but does not have any effect during path construction.

        The default line cap is :obj:`MITER <LINE_JOIN_MITER>`.

        :param line_join: A :ref:`LINE_JOIN` string.

        """
    cairo.cairo_set_line_join(self._pointer, line_join)
    self._check_status()