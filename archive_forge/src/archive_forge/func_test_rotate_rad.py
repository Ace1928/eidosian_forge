import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_rotate_rad(self):
    axis = Vector3(0, 0, 1)
    tests = (((1, 0, 0), math.pi), ((1, 0, 0), math.pi / 2), ((1, 0, 0), -math.pi / 2), ((1, 0, 0), math.pi / 4))
    for initialVec, radians in tests:
        vec = Vector3(initialVec).rotate_rad(radians, axis)
        self.assertEqual(vec, (math.cos(radians), math.sin(radians), 0))