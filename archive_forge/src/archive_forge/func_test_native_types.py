import unittest
from ctypes import *
import re, sys
def test_native_types(self):
    for tp, fmt, shape, itemtp in native_types:
        ob = tp()
        v = memoryview(ob)
        try:
            self.assertEqual(normalize(v.format), normalize(fmt))
            if shape:
                self.assertEqual(len(v), shape[0])
            else:
                self.assertEqual(len(v) * sizeof(itemtp), sizeof(ob))
            self.assertEqual(v.itemsize, sizeof(itemtp))
            self.assertEqual(v.shape, shape)
            self.assertFalse(v.readonly)
            if v.shape:
                n = 1
                for dim in v.shape:
                    n = n * dim
                self.assertEqual(n * v.itemsize, len(v.tobytes()))
        except:
            print(tp)
            raise