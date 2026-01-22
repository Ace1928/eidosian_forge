import cupy
import operator
import numpy
from cupy._core._dtype import get_dtype
def validateaxis(axis):
    if axis is not None:
        axis_type = type(axis)
        if axis_type == tuple:
            raise TypeError("Tuples are not accepted for the 'axis' parameter. Please pass in one of the following: {-2, -1, 0, 1, None}.")
        if not cupy.issubdtype(cupy.dtype(axis_type), cupy.integer):
            raise TypeError('axis must be an integer, not {name}'.format(name=axis_type.__name__))
        if not -2 <= axis <= 1:
            raise ValueError('axis out of range')