import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__both_endpoints_inside(self):
    """Ensures lines that overlap the rect are clipped.

        Testing lines with both endpoints inside the rect.
        """
    rect = Rect((-10, -5), (20, 20))
    small_rect = rect.inflate(-2, -2)
    lines = ((small_rect.midleft, small_rect.midright), (small_rect.midtop, small_rect.midbottom), (small_rect.topleft, small_rect.bottomright), (small_rect.topright, small_rect.bottomleft))
    for line in lines:
        expected_line = line
        clipped_line = rect.clipline(line)
        self.assertTupleEqual(clipped_line, expected_line)
        expected_line = (expected_line[1], expected_line[0])
        clipped_line = rect.clipline((line[1], line[0]))
        self.assertTupleEqual(clipped_line, expected_line)