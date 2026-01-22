import os
import contextlib
import functools
import gc
import socket
import sys
import textwrap
import types
import warnings
def get_contiguous_span(shape, strides, itemsize):
    """
    Return a contiguous span of N-D array data.

    Parameters
    ----------
    shape : tuple
    strides : tuple
    itemsize : int
      Specify array shape data

    Returns
    -------
    start, end : int
      The span end points.
    """
    if not strides:
        start = 0
        end = itemsize * product(shape)
    else:
        start = 0
        end = itemsize
        for i, dim in enumerate(shape):
            if dim == 0:
                start = end = 0
                break
            stride = strides[i]
            if stride > 0:
                end += stride * (dim - 1)
            elif stride < 0:
                start += stride * (dim - 1)
        if end - start != itemsize * product(shape):
            raise ValueError('array data is non-contiguous')
    return (start, end)