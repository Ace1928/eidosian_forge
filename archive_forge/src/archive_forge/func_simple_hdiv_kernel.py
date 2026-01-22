import itertools
import numpy as np
import operator
import re
from numba import cuda, int64
from numba.cuda import compile_ptx
from numba.core.errors import TypingError
from numba.core.types import f2
from numba.cuda.testing import (unittest, CUDATestCase, skip_on_cudasim,
def simple_hdiv_kernel(ary, array_a, array_b):
    i = cuda.grid(1)
    if i < ary.size:
        a = array_a[i]
        b = array_b[i]
        ary[i] = cuda.fp16.hdiv(a, b)