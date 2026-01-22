from ctypes import *
import array
import gc
import unittest
def test_from_buffer_copy(self):
    a = array.array('i', range(16))
    x = (c_int * 16).from_buffer_copy(a)
    y = X.from_buffer_copy(a)
    self.assertEqual(y.c_int, a[0])
    self.assertFalse(y.init_called)
    self.assertEqual(x[:], list(range(16)))
    a[0], a[-1] = (200, -200)
    self.assertEqual(x[:], list(range(16)))
    a.append(100)
    self.assertEqual(x[:], list(range(16)))
    self.assertEqual(x._objects, None)
    del a
    gc.collect()
    gc.collect()
    gc.collect()
    self.assertEqual(x[:], list(range(16)))
    x = (c_char * 16).from_buffer_copy(b'a' * 16)
    self.assertEqual(x[:], b'a' * 16)
    with self.assertRaises(TypeError):
        (c_char * 16).from_buffer_copy('a' * 16)