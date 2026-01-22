import sys, unittest
from ctypes import *
def test_native(self):
    for typ in structures:
        self.assertEqual(typ.value.offset, 1)
        o = typ()
        o.value = 4
        self.assertEqual(o.value, 4)