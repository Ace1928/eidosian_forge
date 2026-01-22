import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
@unittest.skipIf(IS_PYPY, 'known pypy failure')
def test_invalid_swizzle(self):

    def invalidSwizzleX():
        Vector3().xx = (1, 2)

    def invalidSwizzleY():
        Vector3().yy = (1, 2)

    def invalidSwizzleZ():
        Vector3().zz = (1, 2)

    def invalidSwizzleW():
        Vector3().ww = (1, 2)
    self.assertRaises(AttributeError, invalidSwizzleX)
    self.assertRaises(AttributeError, invalidSwizzleY)
    self.assertRaises(AttributeError, invalidSwizzleZ)
    self.assertRaises(AttributeError, invalidSwizzleW)

    def invalidAssignment():
        Vector3().xy = 3
    self.assertRaises(TypeError, invalidAssignment)