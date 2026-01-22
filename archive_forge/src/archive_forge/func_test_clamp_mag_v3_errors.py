import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v3_errors(self):
    v1 = Vector3(1, 2, 2)
    for invalid_args in (('foo', 'bar'), (1, 2, 3), (342.234, 'test')):
        with self.subTest(invalid_args=invalid_args):
            self.assertRaises(TypeError, v1.clamp_magnitude, *invalid_args)
            self.assertRaises(TypeError, v1.clamp_magnitude_ip, *invalid_args)
    for invalid_args in ((-1,), (4, 3), (-4, 10), (-4, -2)):
        with self.subTest(invalid_args=invalid_args):
            self.assertRaises(ValueError, v1.clamp_magnitude, *invalid_args)
            self.assertRaises(ValueError, v1.clamp_magnitude_ip, *invalid_args)
    v2 = Vector3()
    self.assertRaises(ValueError, v2.clamp_magnitude, 3)
    self.assertRaises(ValueError, v2.clamp_magnitude_ip, 4)