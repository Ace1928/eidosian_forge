import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_height(self):
    """Changing the height resizes the rect from the top-left corner"""
    r = Rect(1, 2, 3, 4)
    new_height = 10
    old_topleft = r.topleft
    old_width = r.width
    r.height = new_height
    self.assertEqual(new_height, r.height)
    self.assertEqual(old_width, r.width)
    self.assertEqual(old_topleft, r.topleft)