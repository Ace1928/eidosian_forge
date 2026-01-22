import operator
import torch
from . import _dtypes_impl
def axis_none_flatten(*tensors, axis=None):
    """Flatten the arrays if axis is None."""
    if axis is None:
        tensors = tuple((ar.flatten() for ar in tensors))
        return (tensors, 0)
    else:
        return (tensors, axis)