import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_move_towards_basic(self):
    expected = Vector3(7.93205057, 2006.38284641, 43.8078042)
    origin = Vector3(7.22, 2004.0, 42.13)
    target = Vector3(12.3, 2021.0, 54.1)
    change_ip = origin.copy()
    change = origin.move_towards(target, 3)
    change_ip.move_towards_ip(target, 3)
    self.assertEqual(change, expected)
    self.assertEqual(change_ip, expected)