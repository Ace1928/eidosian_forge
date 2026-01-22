import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_bottomleft(self):
    """Changing the bottomleft attribute moves the rect and does not change
        the rect's size
        """
    r = Rect(1, 2, 3, 4)
    new_bottomleft = (r.left + 20, r.bottom + 30)
    expected_topleft = (r.left + 20, r.top + 30)
    old_size = r.size
    r.bottomleft = new_bottomleft
    self.assertEqual(new_bottomleft, r.bottomleft)
    self.assertEqual(expected_topleft, r.topleft)
    self.assertEqual(old_size, r.size)