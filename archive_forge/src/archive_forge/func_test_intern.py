import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test_intern(self):
    unique_str1 = 'unique str ' + osutils.rand_chars(20)
    unique_str2 = 'unique str ' + osutils.rand_chars(20)
    key = self.module.StaticTuple(unique_str1, unique_str2)
    self.assertFalse(key in self.module._interned_tuples)
    key2 = self.module.StaticTuple(unique_str1, unique_str2)
    self.assertEqual(key, key2)
    self.assertIsNot(key, key2)
    key3 = key.intern()
    self.assertIs(key, key3)
    self.assertTrue(key in self.module._interned_tuples)
    self.assertEqual(key, self.module._interned_tuples[key])
    key2 = key2.intern()
    self.assertIs(key, key2)