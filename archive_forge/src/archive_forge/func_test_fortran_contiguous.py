from ctypes import *
import array
import gc
import unittest
def test_fortran_contiguous(self):
    try:
        import _testbuffer
    except ImportError as err:
        self.skipTest(str(err))
    flags = _testbuffer.ND_WRITABLE | _testbuffer.ND_FORTRAN
    array = _testbuffer.ndarray([97] * 16, format='B', shape=[4, 4], flags=flags)
    with self.assertRaisesRegex(TypeError, 'not C contiguous'):
        (c_char * 16).from_buffer(array)
    array = memoryview(array)
    self.assertTrue(array.f_contiguous)
    self.assertFalse(array.c_contiguous)
    with self.assertRaisesRegex(TypeError, 'not C contiguous'):
        (c_char * 16).from_buffer(array)