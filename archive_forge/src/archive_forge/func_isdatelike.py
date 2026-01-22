import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def isdatelike(ds):
    """ Has a date or datetime measure

    >>> isdatelike('int32')
    False
    >>> isdatelike('3 * datetime')
    True
    >>> isdatelike('?datetime')
    True
    """
    ds = launder(ds)
    return ds == date_ or ds == datetime_