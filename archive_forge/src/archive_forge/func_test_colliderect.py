import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_colliderect(self):
    r1 = Rect(1, 2, 3, 4)
    self.assertTrue(r1.colliderect(Rect(0, 0, 2, 3)), 'r1 does not collide with Rect(0, 0, 2, 3)')
    self.assertFalse(r1.colliderect(Rect(0, 0, 1, 2)), 'r1 collides with Rect(0, 0, 1, 2)')
    self.assertFalse(r1.colliderect(Rect(r1.right, r1.bottom, 2, 2)), 'r1 collides with Rect(r1.right, r1.bottom, 2, 2)')
    self.assertTrue(r1.colliderect(Rect(r1.left + 1, r1.top + 1, r1.width - 2, r1.height - 2)), 'r1 does not collide with Rect(r1.left + 1, r1.top + 1, ' + 'r1.width - 2, r1.height - 2)')
    self.assertTrue(r1.colliderect(Rect(r1.left - 1, r1.top - 1, r1.width + 2, r1.height + 2)), 'r1 does not collide with Rect(r1.left - 1, r1.top - 1, ' + 'r1.width + 2, r1.height + 2)')
    self.assertTrue(r1.colliderect(Rect(r1)), 'r1 does not collide with an identical rect')
    self.assertFalse(r1.colliderect(Rect(r1.right, r1.bottom, 0, 0)), 'r1 collides with Rect(r1.right, r1.bottom, 0, 0)')
    self.assertFalse(r1.colliderect(Rect(r1.right, r1.bottom, 1, 1)), 'r1 collides with Rect(r1.right, r1.bottom, 1, 1)')