from __future__ import annotations
from functools import partial
import numpy as np
from xarray.coding.variables import (
from xarray.core import indexing
from xarray.core.utils import module_available
from xarray.core.variable import Variable
from xarray.namedarray.parallelcompat import get_chunked_array_type
from xarray.namedarray.pycompat import is_chunked_array
def char_to_bytes(arr):
    """Convert numpy/dask arrays from characters to fixed width bytes."""
    if arr.dtype != 'S1':
        raise ValueError("argument must have dtype='S1'")
    if not arr.ndim:
        return arr
    size = arr.shape[-1]
    if not size:
        return np.zeros(arr.shape[:-1], dtype=np.bytes_)
    if is_chunked_array(arr):
        chunkmanager = get_chunked_array_type(arr)
        if len(arr.chunks[-1]) > 1:
            raise ValueError(f'cannot stacked dask character array with multiple chunks in the last dimension: {arr}')
        dtype = np.dtype('S' + str(arr.shape[-1]))
        return chunkmanager.map_blocks(_numpy_char_to_bytes, arr, dtype=dtype, chunks=arr.chunks[:-1], drop_axis=[arr.ndim - 1])
    else:
        return StackedBytesArray(arr)