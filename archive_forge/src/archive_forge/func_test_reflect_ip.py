import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_reflect_ip(self):
    v1 = Vector3(1, -1, 1)
    v2 = Vector3(v1)
    n = Vector3(0, 1, 0)
    self.assertEqual(v2.reflect_ip(n), None)
    self.assertEqual(v2, Vector3(1, 1, 1))
    v2 = Vector3(v1)
    v2.reflect_ip(3 * n)
    self.assertEqual(v2, v1.reflect(n))
    v2 = Vector3(v1)
    v2.reflect_ip(-v1)
    self.assertEqual(v2, -v1)
    self.assertRaises(ValueError, lambda: v2.reflect_ip(self.zeroVec))