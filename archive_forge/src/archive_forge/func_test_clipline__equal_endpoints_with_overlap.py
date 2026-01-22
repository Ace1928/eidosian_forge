import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__equal_endpoints_with_overlap(self):
    """Ensures clipline handles lines with both endpoints the same.

        Testing lines that overlap the rect.
        """
    rect = Rect((10, 25), (15, 20))
    pts = ((x, y) for x in range(rect.left, rect.right) for y in range(rect.top, rect.bottom))
    for pt in pts:
        expected_line = (pt, pt)
        clipped_line = rect.clipline((pt, pt))
        self.assertTupleEqual(clipped_line, expected_line)