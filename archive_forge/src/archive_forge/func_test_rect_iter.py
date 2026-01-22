import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_rect_iter(self):
    rect = Rect(50, 100, 150, 200)
    rect_iterator = rect.__iter__()
    for i, val in enumerate(rect_iterator):
        self.assertEqual(rect[i], val)