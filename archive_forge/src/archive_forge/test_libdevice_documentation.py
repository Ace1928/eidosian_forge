import numpy as np
from numba.core import types
from numba.cuda.testing import skip_on_cudasim, unittest, CUDATestCase
from numba import cuda
from numba.cuda import libdevice, compile_ptx
from numba.cuda.libdevicefuncs import functions, create_signature
from numba.cuda import libdevice

    Class for holding all tests of compiling calls to libdevice functions. We
    generate the actual tests in this class (as opposed to using subTest and
    one test within this class) because there are a lot of tests, and it makes
    the test suite appear frozen to test them all as subTests in one test.
    