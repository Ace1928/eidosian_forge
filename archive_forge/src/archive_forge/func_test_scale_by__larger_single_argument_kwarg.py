import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_scale_by__larger_single_argument_kwarg(self):
    """The scale method scales around the center of the rectangle using
        keyword arguments 'x' and 'y'"""
    r = Rect(2, 4, 6, 8)
    r2 = r.scale_by(x=2)
    self.assertEqual(r.center, r2.center)
    self.assertEqual(r.left - 3, r2.left)
    self.assertEqual(r.top - 4, r2.top)
    self.assertEqual(r.right + 3, r2.right)
    self.assertEqual(r.bottom + 4, r2.bottom)
    self.assertEqual(r.width * 2, r2.width)
    self.assertEqual(r.height * 2, r2.height)