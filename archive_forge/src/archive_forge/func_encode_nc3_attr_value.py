from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
def encode_nc3_attr_value(value):
    if isinstance(value, bytes):
        pass
    elif isinstance(value, str):
        value = value.encode(STRING_ENCODING)
    else:
        value = coerce_nc3_dtype(np.atleast_1d(value))
        if value.ndim > 1:
            raise ValueError('netCDF attributes must be 1-dimensional')
    return value