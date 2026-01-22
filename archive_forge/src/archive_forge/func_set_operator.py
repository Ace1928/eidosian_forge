from . import _check_status, _keepref, cairo, constants, ffi
from .fonts import FontFace, FontOptions, ScaledFont, _encode_string
from .matrix import Matrix
from .patterns import Pattern
from .surfaces import Surface
def set_operator(self, operator):
    """Set the current :ref:`OPERATOR`
        to be used for all drawing operations.

        The default operator is :obj:`OVER <OPERATOR_OVER>`.

        :param operator: A :ref:`OPERATOR` string.

        """
    cairo.cairo_set_operator(self._pointer, operator)
    self._check_status()