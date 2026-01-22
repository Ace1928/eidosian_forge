import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_inflate_ip__larger(self):
    """The inflate_ip method inflates around the center of the rectangle"""
    r = Rect(2, 4, 6, 8)
    r2 = Rect(r)
    r2.inflate_ip(-4, -6)
    self.assertEqual(r.center, r2.center)
    self.assertEqual(r.left + 2, r2.left)
    self.assertEqual(r.top + 3, r2.top)
    self.assertEqual(r.right - 2, r2.right)
    self.assertEqual(r.bottom - 3, r2.bottom)
    self.assertEqual(r.width - 4, r2.width)
    self.assertEqual(r.height - 6, r2.height)