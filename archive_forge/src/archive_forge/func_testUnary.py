import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testUnary(self):
    v = +self.v1
    self.assertTrue(isinstance(v, type(self.v1)))
    self.assertEqual(v.x, self.v1.x)
    self.assertEqual(v.y, self.v1.y)
    self.assertEqual(v.z, self.v1.z)
    self.assertNotEqual(id(v), id(self.v1))
    v = -self.v1
    self.assertTrue(isinstance(v, type(self.v1)))
    self.assertEqual(v.x, -self.v1.x)
    self.assertEqual(v.y, -self.v1.y)
    self.assertEqual(v.z, -self.v1.z)
    self.assertNotEqual(id(v), id(self.v1))