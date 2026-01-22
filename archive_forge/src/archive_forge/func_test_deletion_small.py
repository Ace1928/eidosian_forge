import ctypes
import random
from numba.tests.support import TestCase
from numba import _helperlib, jit, typed, types
from numba.core.config import IS_32BITS
from numba.core.datamodel.models import UniTupleModel
from numba.extending import register_model, typeof_impl, unbox, overload
def test_deletion_small(self):
    d = Dict(self, 4, 8)
    self.assertEqual(len(d), 0)
    self.assertIsNone(d.get('abcd'))
    d['abcd'] = 'cafe0000'
    d['abce'] = 'cafe0001'
    d['abcf'] = 'cafe0002'
    self.assertEqual(len(d), 3)
    self.assertEqual(d['abcd'], 'cafe0000')
    self.assertEqual(d['abce'], 'cafe0001')
    self.assertEqual(d['abcf'], 'cafe0002')
    self.assertEqual(len(d), 3)
    del d['abcd']
    self.assertIsNone(d.get('abcd'))
    self.assertEqual(d['abce'], 'cafe0001')
    self.assertEqual(d['abcf'], 'cafe0002')
    self.assertEqual(len(d), 2)
    with self.assertRaises(KeyError):
        del d['abcd']
    del d['abcf']
    self.assertIsNone(d.get('abcd'))
    self.assertEqual(d['abce'], 'cafe0001')
    self.assertIsNone(d.get('abcf'))
    self.assertEqual(len(d), 1)
    del d['abce']
    self.assertIsNone(d.get('abcd'))
    self.assertIsNone(d.get('abce'))
    self.assertIsNone(d.get('abcf'))
    self.assertEqual(len(d), 0)