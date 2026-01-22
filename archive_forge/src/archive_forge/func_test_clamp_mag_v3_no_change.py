import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v3_no_change(self):
    v1 = Vector3(1, 2, 3)
    for args in ((1, 6), (1.12, 5.55), (0.93, 6.83), (7.6,)):
        with self.subTest(args=args):
            v2 = v1.clamp_magnitude(*args)
            v1.clamp_magnitude_ip(*args)
            self.assertEqual(v1, v2)
            self.assertEqual(v1, Vector3(1, 2, 3))