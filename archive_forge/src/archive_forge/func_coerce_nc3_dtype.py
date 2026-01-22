from __future__ import annotations
import unicodedata
import numpy as np
from xarray import coding
from xarray.core.variable import Variable
def coerce_nc3_dtype(arr):
    """Coerce an array to a data type that can be stored in a netCDF-3 file

    This function performs the dtype conversions as specified by the
    ``_nc3_dtype_coercions`` mapping:
        int64  -> int32
        uint64 -> int32
        uint32 -> int32
        uint16 -> int16
        uint8  -> int8
        bool   -> int8

    Data is checked for equality, or equivalence (non-NaN values) using the
    ``(cast_array == original_array).all()``.
    """
    dtype = str(arr.dtype)
    if dtype in _nc3_dtype_coercions:
        new_dtype = _nc3_dtype_coercions[dtype]
        cast_arr = arr.astype(new_dtype)
        if not (cast_arr == arr).all():
            raise ValueError(COERCION_VALUE_ERROR.format(dtype=dtype, new_dtype=new_dtype))
        arr = cast_arr
    return arr