import operator
import pickle
import sys
from breezy import debug, osutils, tests
from breezy.bzr import _static_tuple_py, static_tuple
from breezy.tests import features
def test__c_intern_handles_refcount(self):
    if self.module is _static_tuple_py:
        return
    unique_str1 = 'unique str ' + osutils.rand_chars(20)
    unique_str2 = 'unique str ' + osutils.rand_chars(20)
    key = self.module.StaticTuple(unique_str1, unique_str2)
    self.assertRefcount(1, key)
    self.assertFalse(key in self.module._interned_tuples)
    self.assertFalse(key._is_interned())
    key2 = self.module.StaticTuple(unique_str1, unique_str2)
    self.assertRefcount(1, key)
    self.assertRefcount(1, key2)
    self.assertEqual(key, key2)
    self.assertIsNot(key, key2)
    key3 = key.intern()
    self.assertIs(key, key3)
    self.assertTrue(key in self.module._interned_tuples)
    self.assertEqual(key, self.module._interned_tuples[key])
    self.assertRefcount(2, key)
    del key3
    self.assertRefcount(1, key)
    self.assertTrue(key._is_interned())
    self.assertRefcount(1, key2)
    key3 = key2.intern()
    self.assertRefcount(2, key)
    self.assertRefcount(1, key2)
    self.assertIs(key, key3)
    self.assertIsNot(key3, key2)
    del key2
    del key3
    self.assertRefcount(1, key)