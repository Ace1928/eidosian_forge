import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_ufunc_props_jit(self):
    duadd = self.nopython_dufunc(pyuadd)
    duadd(1, 2)
    attributes = {'nin': duadd.nin, 'nout': duadd.nout, 'nargs': duadd.nargs, 'identity': duadd.identity, 'signature': duadd.signature}

    def get_attr_fn(attr):
        fn = f'\n                def impl():\n                    return duadd.{attr}\n            '
        l = {}
        exec(textwrap.dedent(fn), {'duadd': duadd}, l)
        return l['impl']
    for attr, val in attributes.items():
        cfunc = njit(get_attr_fn(attr))
        self.assertEqual(val, cfunc(), f'Attribute differs from original: {attr}')