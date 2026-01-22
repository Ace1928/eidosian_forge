import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_move_towards_max_distance(self):
    expected = Vector3(12.3, 2021, 42.5)
    origin = Vector3(7.22, 2004.0, 17.5)
    change_ip = origin.copy()
    change = origin.move_towards(expected, 100)
    change_ip.move_towards_ip(expected, 100)
    self.assertEqual(change, expected)
    self.assertEqual(change_ip, expected)