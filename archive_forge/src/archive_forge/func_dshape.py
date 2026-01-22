from __future__ import print_function, division, absolute_import
from itertools import chain
import operator
from .. import parser
from .. import type_symbol_table
from ..validation import validate
from .. import coretypes
def dshape(o):
    """
    Parse a datashape. For a thorough description see
    http://blaze.pydata.org/docs/datashape.html

    >>> ds = dshape('2 * int32')
    >>> ds[1]
    ctype("int32")
    """
    if isinstance(o, coretypes.DataShape):
        return o
    if isinstance(o, str):
        ds = parser.parse(o, type_symbol_table.sym)
    elif isinstance(o, (coretypes.CType, coretypes.String, coretypes.Record, coretypes.JSON, coretypes.Date, coretypes.Time, coretypes.DateTime, coretypes.Unit)):
        ds = coretypes.DataShape(o)
    elif isinstance(o, coretypes.Mono):
        ds = o
    elif isinstance(o, (list, tuple)):
        ds = coretypes.DataShape(*o)
    else:
        raise TypeError('Cannot create dshape from object of type %s' % type(o))
    validate(ds)
    return ds