import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_move_nowhere(self):
    origin = Vector3(7.22, 2004.0, 24.5)
    target = Vector3(12.3, 2021.0, 3.2)
    change_ip = origin.copy()
    change = origin.move_towards(target, 0)
    change_ip.move_towards_ip(target, 0)
    self.assertEqual(change, origin)
    self.assertEqual(change_ip, origin)