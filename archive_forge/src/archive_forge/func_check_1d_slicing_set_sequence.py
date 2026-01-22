import decimal
import itertools
import numpy as np
import unittest
from numba import jit, njit, typeof
from numba.core import utils, types, errors
from numba.tests.support import TestCase, tag
from numba.core.typing import arraydecl
from numba.core.types import intp, ellipsis, slice2_type, slice3_type
def check_1d_slicing_set_sequence(self, flags, seqty, seq):
    """
        Generic sequence to 1d slice assignment
        """
    pyfunc = slicing_1d_usecase_set
    dest_type = types.Array(types.int32, 1, 'C')
    argtys = (dest_type, seqty, types.int32, types.int32, types.int32)
    cfunc = jit(argtys, **flags)(pyfunc).overloads[argtys].entry_point
    N = 10
    k = len(seq)
    arg = np.arange(N, dtype=np.int32)
    args = (seq, 1, -N + k + 1, 1)
    expected = pyfunc(arg.copy(), *args)
    got = cfunc(arg.copy(), *args)
    self.assertPreciseEqual(expected, got)
    args = (seq, 1, -N + k, 1)
    with self.assertRaises(ValueError) as raises:
        cfunc(arg.copy(), *args)