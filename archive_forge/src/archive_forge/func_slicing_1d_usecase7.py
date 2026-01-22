import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def slicing_1d_usecase7(a, start):
    b = a[start::-2]
    total = 0
    for i in range(b.shape[0]):
        total += b[i] * (i + 1)
    return total