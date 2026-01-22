import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_update__RectObject(self):
    """Test update with other rect object"""
    rect = Rect(0, 0, 1, 1)
    rect2 = Rect(1, 2, 3, 4)
    rect.update(rect2)
    self.assertEqual(1, rect.left)
    self.assertEqual(2, rect.top)
    self.assertEqual(3, rect.width)
    self.assertEqual(4, rect.height)