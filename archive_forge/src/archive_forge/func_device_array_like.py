from contextlib import contextmanager
import numpy as np
from_record_like = None
def device_array_like(ary, stream=0):
    strides = _contiguous_strides_like_array(ary)
    order = _order_like_array(ary)
    return device_array(shape=ary.shape, dtype=ary.dtype, strides=strides, order=order)