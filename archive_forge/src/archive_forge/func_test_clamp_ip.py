import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clamp_ip(self):
    r = Rect(10, 10, 10, 10)
    c = Rect(19, 12, 5, 5)
    c.clamp_ip(r)
    self.assertEqual(c.right, r.right)
    self.assertEqual(c.top, 12)
    c = Rect(1, 2, 3, 4)
    c.clamp_ip(r)
    self.assertEqual(c.topleft, r.topleft)
    c = Rect(5, 500, 22, 33)
    c.clamp_ip(r)
    self.assertEqual(c.center, r.center)