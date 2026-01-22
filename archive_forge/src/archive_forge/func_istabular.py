import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def istabular(ds):
    """ A collection of records

    >>> istabular('var * {name: string, amount: int}')
    True
    >>> istabular('var * 10 * 3 * int')
    False
    >>> istabular('10 * var * int')
    False
    >>> istabular('var * (int64, string, ?float64)')
    False
    """
    ds = dshape(ds)
    return _dimensions(ds) == 1 and isrecord(ds.measure)