import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_move_ip(self):
    r = Rect(1, 2, 3, 4)
    r2 = Rect(r)
    move_x = 10
    move_y = 20
    r2.move_ip(move_x, move_y)
    expected_r2 = Rect(r.left + move_x, r.top + move_y, r.width, r.height)
    self.assertEqual(expected_r2, r2)