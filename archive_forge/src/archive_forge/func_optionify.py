from __future__ import absolute_import
import numpy as np
from datashader import datashape
def optionify(lhs, rhs, dshape):
    """Check whether a binary operation's dshape came from
    :class:`~datashape.coretypes.Option` typed operands and construct an
    :class:`~datashape.coretypes.Option` type accordingly.

    Examples
    --------
    >>> from datashader.datashape import int32, int64, Option
    >>> x = Option(int32)
    >>> x
    Option(ty=ctype("int32"))
    >>> y = int64
    >>> y
    ctype("int64")
    >>> optionify(x, y, int64)
    Option(ty=ctype("int64"))
    """
    if hasattr(dshape.measure, 'ty'):
        return dshape
    if hasattr(lhs, 'ty') or hasattr(rhs, 'ty'):
        return datashape.Option(dshape)
    return dshape