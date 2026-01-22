import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_normalize__positive_height(self):
    """Ensures normalize works with a negative width and a positive height."""
    test_rect = Rect((1, 2), (-3, 6))
    expected_normalized_rect = ((test_rect.x + test_rect.w, test_rect.y), (-test_rect.w, test_rect.h))
    test_rect.normalize()
    self.assertEqual(test_rect, expected_normalized_rect)