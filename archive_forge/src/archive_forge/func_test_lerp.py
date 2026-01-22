import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_lerp(self):
    v1 = Vector3(0, 0, 0)
    v2 = Vector3(10, 10, 10)
    self.assertEqual(v1.lerp(v2, 0.5), (5, 5, 5))
    self.assertRaises(ValueError, lambda: v1.lerp(v2, 2.5))
    v1 = Vector3(-10, -5, -20)
    v2 = Vector3(10, 10, -20)
    self.assertEqual(v1.lerp(v2, 0.5), (0, 2.5, -20))