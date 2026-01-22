import numpy as np
from .util import collect, dshape
from .internal_utils import remove
from .coretypes import (DataShape, Fixed, Var, Ellipsis, Record, Unit,
from .typesets import floating, boolean
def isrecord(ds):
    """ Is this dshape a record type?

    >>> isrecord('{name: string, amount: int}')
    True
    >>> isrecord('int')
    False
    >>> isrecord('?{name: string, amount: int}')
    True
    """
    if isinstance(ds, str):
        ds = dshape(ds)
    if isinstance(ds, DataShape) and len(ds) == 1:
        ds = ds[0]
    return isinstance(getattr(ds, 'ty', ds), Record)