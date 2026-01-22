import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_clamp_mag_v3_min(self):
    v1 = Vector3(3, 1, 2)
    v2 = v1.clamp_magnitude(5, 10)
    v1.clamp_magnitude_ip(5, 10)
    expected_v2 = Vector3(4.008918628686366, 1.3363062095621219, 2.6726124191242437)
    self.assertEqual(expected_v2, v1)
    self.assertEqual(expected_v2, v2)