import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testExtendedSlicing(self):

    def delSlice(vec, start=None, stop=None, step=None):
        if start is not None and stop is not None and (step is not None):
            del vec[start:stop:step]
        elif start is not None and stop is None and (step is not None):
            del vec[start::step]
        elif start is None and stop is None and (step is not None):
            del vec[::step]
    v = Vector3(self.v1)
    self.assertRaises(TypeError, delSlice, v, None, None, 2)
    self.assertRaises(TypeError, delSlice, v, 1, None, 2)
    self.assertRaises(TypeError, delSlice, v, 1, 2, 1)
    v = Vector3(self.v1)
    v[::2] = [-1.1, -2.2]
    self.assertEqual(v, [-1.1, self.v1.y, -2.2])
    v = Vector3(self.v1)
    v[::-2] = [10, 20]
    self.assertEqual(v, [20, self.v1.y, 10])
    v = Vector3(self.v1)
    v[::-1] = v
    self.assertEqual(v, [self.v1.z, self.v1.y, self.v1.x])
    a = Vector3(self.v1)
    b = Vector3(self.v1)
    c = Vector3(self.v1)
    a[1:2] = [2.2]
    b[slice(1, 2)] = [2.2]
    c[1:2] = (2.2,)
    self.assertEqual(a, b)
    self.assertEqual(a, c)
    self.assertEqual(type(a), type(self.v1))
    self.assertEqual(type(b), type(self.v1))
    self.assertEqual(type(c), type(self.v1))