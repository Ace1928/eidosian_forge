import numpy as np
def prepare_shape_strides_dtype(shape, strides, dtype, order):
    dtype = np.dtype(dtype)
    if isinstance(shape, int):
        shape = (shape,)
    if isinstance(strides, int):
        strides = (strides,)
    else:
        strides = strides or _fill_stride_by_order(shape, dtype, order)
    return (shape, strides, dtype)