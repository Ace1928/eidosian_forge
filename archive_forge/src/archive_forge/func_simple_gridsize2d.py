import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def simple_gridsize2d(ary):
    i, j = cuda.grid(2)
    x, y = cuda.gridsize(2)
    if i == 0 and j == 0:
        ary[0] = x
        ary[1] = y