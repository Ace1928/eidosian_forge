import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collidelistall_list_of_tuples(self):
    r = Rect(1, 1, 10, 10)
    l = [(1, 1, 10, 10), (5, 5, 10, 10), (15, 15, 1, 1), (2, 2, 1, 1)]
    self.assertEqual(r.collidelistall(l), [0, 1, 3])
    f = [(50, 50, 1, 1), (20, 20, 5, 5)]
    self.assertFalse(r.collidelistall(f))