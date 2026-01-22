import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def iscollection(ds):
    """ Is a collection of items, has dimension

    >>> iscollection('5 * int32')
    True
    >>> iscollection('int32')
    False
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    return isdimension(ds[0])