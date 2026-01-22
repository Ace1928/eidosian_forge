import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidepoint(self):
    r = Rect(1, 2, 3, 4)
    self.assertTrue(r.collidepoint(r.left, r.top), 'r does not collide with point (left, top)')
    self.assertFalse(r.collidepoint(r.left - 1, r.top), 'r collides with point (left - 1, top)')
    self.assertFalse(r.collidepoint(r.left, r.top - 1), 'r collides with point (left, top - 1)')
    self.assertFalse(r.collidepoint(r.left - 1, r.top - 1), 'r collides with point (left - 1, top - 1)')
    self.assertTrue(r.collidepoint(r.right - 1, r.bottom - 1), 'r does not collide with point (right - 1, bottom - 1)')
    self.assertFalse(r.collidepoint(r.right, r.bottom), 'r collides with point (right, bottom)')
    self.assertFalse(r.collidepoint(r.right - 1, r.bottom), 'r collides with point (right - 1, bottom)')
    self.assertFalse(r.collidepoint(r.right, r.bottom - 1), 'r collides with point (right, bottom - 1)')