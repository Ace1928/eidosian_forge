import numpy
from numpy import linalg
import warnings
import cupy
from cupy import _core
from cupy_backends.cuda.libs import cublas
from cupy.cuda import device
from cupy.linalg import _util
def get_batched_gesv_limit():
    global _batched_gesv_limit
    return _batched_gesv_limit