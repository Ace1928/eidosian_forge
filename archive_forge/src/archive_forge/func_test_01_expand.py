from __future__ import with_statement, division
from functools import partial
from passlib.utils import getrandbytes
from passlib.tests.utils import TestCase
def test_01_expand(self):
    """expand_des_key()"""
    from passlib.crypto.des import expand_des_key, shrink_des_key, _KDATA_MASK, INT_56_MASK
    for key1, _, _ in self.des_test_vectors:
        key2 = shrink_des_key(key1)
        key3 = expand_des_key(key2)
        self.assertEqual(key3, key1 & _KDATA_MASK)
    self.assertRaises(TypeError, expand_des_key, 1.0)
    self.assertRaises(ValueError, expand_des_key, INT_56_MASK + 1)
    self.assertRaises(ValueError, expand_des_key, b'\x00' * 8)
    self.assertRaises(ValueError, expand_des_key, -1)
    self.assertRaises(ValueError, expand_des_key, b'\x00' * 6)