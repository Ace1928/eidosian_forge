import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice
def use_sincos(s, c, x):
    i = cuda.grid(1)
    if i < len(x):
        sr, cr = libdevice.sincos(x[i])
        s[i] = sr
        c[i] = cr