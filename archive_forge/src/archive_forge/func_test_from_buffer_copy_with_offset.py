from ctypes import *
import array
import gc
import unittest
def test_from_buffer_copy_with_offset(self):
    a = array.array('i', range(16))
    x = (c_int * 15).from_buffer_copy(a, sizeof(c_int))
    self.assertEqual(x[:], a.tolist()[1:])
    with self.assertRaises(ValueError):
        c_int.from_buffer_copy(a, -1)
    with self.assertRaises(ValueError):
        (c_int * 16).from_buffer_copy(a, sizeof(c_int))
    with self.assertRaises(ValueError):
        (c_int * 1).from_buffer_copy(a, 16 * sizeof(c_int))