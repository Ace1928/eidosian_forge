import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_move_away(self):
    expected = Vector3(6.74137906, 2002.39831577, 49.70890994)
    origin = Vector3(7.22, 2004.0, 52.2)
    target = Vector3(12.3, 2021.0, 78.64)
    change_ip = origin.copy()
    change = origin.move_towards(target, -3)
    change_ip.move_towards_ip(target, -3)
    self.assertEqual(change, expected)
    self.assertEqual(change_ip, expected)