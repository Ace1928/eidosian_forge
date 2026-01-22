import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__floats_kwargs(self):
    """Ensures clipline handles four float parameters.

        Tests the clipline(x1, y1, x2, y2) format.
        """
    rect = Rect((1, 2), (35, 40))
    x1 = 5.9
    y1 = 6.9
    x2 = 11.9
    y2 = 19.9
    expected_line = ((math.floor(x1), math.floor(y1)), (math.floor(x2), math.floor(y2)))
    clipped_line = rect.clipline(x1=x1, x2=y1, x3=x2, x4=y2)
    self.assertIsInstance(clipped_line, tuple)
    self.assertTupleEqual(clipped_line, expected_line)