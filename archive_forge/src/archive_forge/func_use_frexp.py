import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice
def use_frexp(frac, exp, x):
    i = cuda.grid(1)
    if i < len(x):
        fracr, expr = libdevice.frexp(x[i])
        frac[i] = fracr
        exp[i] = expr