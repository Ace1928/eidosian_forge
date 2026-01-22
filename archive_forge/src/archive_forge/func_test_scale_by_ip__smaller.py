import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by_ip__smaller(self):
    """The scale method scales around the center of the rectangle"""
    r = Rect(2, 4, 8, 8)
    r2 = Rect(r)
    r2.scale_by_ip(0.5)
    self.assertEqual(r.center, r2.center)
    self.assertEqual(r.left + 2, r2.left)
    self.assertEqual(r.top + 2, r2.top)
    self.assertEqual(r.right - 2, r2.right)
    self.assertEqual(r.bottom - 2, r2.bottom)
    self.assertEqual(r.width / 2, r2.width)
    self.assertEqual(r.height / 2, r2.height)