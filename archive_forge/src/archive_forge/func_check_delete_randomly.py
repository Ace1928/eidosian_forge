import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def check_delete_randomly(self, nmax, ndrop, nrefill, seed=0):
    random.seed(seed)
    d = Dict(self, 8, 8)
    keys = {}

    def make_key(v):
        return 'k_{:06x}'.format(v)

    def make_val(v):
        return 'v_{:06x}'.format(v)
    for i in range(nmax):
        d[make_key(i)] = make_val(i)
    for i in range(nmax):
        k = make_key(i)
        v = make_val(i)
        keys[k] = v
        self.assertEqual(d[k], v)
    self.assertEqual(len(d), nmax)
    droplist = random.sample(list(keys), ndrop)
    remain = keys.copy()
    for i, k in enumerate(droplist, start=1):
        del d[k]
        del remain[k]
        self.assertEqual(len(d), nmax - i)
    self.assertEqual(len(d), nmax - ndrop)
    for k in droplist:
        self.assertIsNone(d.get(k))
    for k in remain:
        self.assertEqual(d[k], remain[k])
    for i in range(nrefill):
        k = make_key(nmax + i)
        v = make_val(nmax + i)
        remain[k] = v
        d[k] = v
    self.assertEqual(len(remain), len(d))
    for k in remain:
        self.assertEqual(d[k], remain[k])