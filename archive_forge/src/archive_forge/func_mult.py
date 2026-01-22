import cmath
import numpy as np
from numba import float32
from numba.types import unicode_type, i8
from numba.pycc import CC, exportmany, export
from numba.tests.support import has_blas
from numba import typed
@cc.export('multf', (float32, float32))
@cc.export('multi', 'i4(i4, i4)')
def mult(a, b):
    return a * b