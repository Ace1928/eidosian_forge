import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_iter_items(self):
    d = Dict(self, 4, 4)
    nmax = 1000

    def make_key(v):
        return '{:04}'.format(v)

    def make_val(v):
        return '{:04}'.format(v + nmax)
    for i in range(nmax):
        d[make_key(i)] = make_val(i)
    for i, (k, v) in enumerate(d.items()):
        self.assertEqual(make_key(i), k)
        self.assertEqual(make_val(i), v)