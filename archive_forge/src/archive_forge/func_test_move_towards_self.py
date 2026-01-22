import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_move_towards_self(self):
    vec = Vector3(6.36, 2001.13, -123.14)
    vec2 = vec.copy()
    for dist in (-3.54, -1, 0, 0.234, 12):
        self.assertEqual(vec.move_towards(vec2, dist), vec)
        vec2.move_towards_ip(vec, dist)
        self.assertEqual(vec, vec2)