import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_rotate_x_rad(self):
    vec = Vector3(0, 1, 0)
    result = vec.rotate_x_rad(math.pi / 2)
    self.assertEqual(result, (0, 0, 1))