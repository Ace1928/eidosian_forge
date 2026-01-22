import cupy
import operator
import numpy
from cupy._core._dtype import get_dtype
def get_index_dtype(arrays=(), maxval=None, check_contents=False):
    """Based on input (integer) arrays ``a``, determines a suitable index data
    type that can hold the data in the arrays.

    Args:
        arrays (tuple of array_like):
            Input arrays whose types/contents to check
        maxval (float, optional):
            Maximum value needed
        check_contents (bool, optional):
            Whether to check the values in the arrays and not just their types.
            Default: False (check only the types)

    Returns:
        dtype: Suitable index data type (int32 or int64)
    """
    int32min = cupy.iinfo(cupy.int32).min
    int32max = cupy.iinfo(cupy.int32).max
    dtype = cupy.int32
    if maxval is not None:
        if maxval > int32max:
            dtype = cupy.int64
    if isinstance(arrays, cupy.ndarray):
        arrays = (arrays,)
    for arr in arrays:
        arr = cupy.asarray(arr)
        if not cupy.can_cast(arr.dtype, cupy.int32):
            if check_contents:
                if arr.size == 0:
                    continue
                elif cupy.issubdtype(arr.dtype, cupy.integer):
                    maxval = arr.max()
                    minval = arr.min()
                    if minval >= int32min and maxval <= int32max:
                        continue
            dtype = cupy.int64
            break
    return dtype