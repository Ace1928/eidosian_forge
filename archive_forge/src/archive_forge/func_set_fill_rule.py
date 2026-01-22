from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_fill_rule(self, fill_rule):
    """Set the current :ref:`FILL_RULE` within the cairo context.
        The fill rule is used to determine which regions are inside
        or outside a complex (potentially self-intersecting) path.
        The current fill rule affects both :meth:`fill` and :meth:`clip`.

        The default fill rule is :obj:`WINDING <FILL_RULE_WINDING>`.

        :param fill_rule: A :ref:`FILL_RULE` string.

        """
    cairo.cairo_set_fill_rule(self._pointer, fill_rule)
    self._check_status()