import os
import numpy
from numpy import linalg
import cupy
import cupy._util
from cupy import _core
import cupyx
def stacked_identity_like(x):
    """
    Precondition: ``x`` is `cupy.ndarray` of shape ``(..., N, N)``
    """
    n = x.shape[-1]
    idx = cupy.arange(n)
    x = cupy.zeros_like(x)
    x[..., idx, idx] = 1
    return x