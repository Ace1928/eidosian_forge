import cmath
import numpy as np
from numba import float32
from numba.types import unicode_type, i8
from numba.pycc import CC, exportmany, export
from numba.tests.support import has_blas
from numba import typed
@cc_nrt.export('dict_usecase', 'intp[:](intp[:])')
def dict_usecase(arr):
    d = typed.Dict()
    for i in range(arr.size):
        d[i] = arr[i]
    out = np.zeros_like(arr)
    for k, v in d.items():
        out[k] = k * v
    return out