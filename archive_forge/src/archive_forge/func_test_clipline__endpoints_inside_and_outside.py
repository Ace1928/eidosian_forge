import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__endpoints_inside_and_outside(self):
    """Ensures lines that overlap the rect are clipped.

        Testing lines with one endpoint outside the rect and the other is
        inside the rect.
        """
    rect = Rect((0, 0), (21, 21))
    big_rect = rect.inflate(2, 2)
    line_dict = {(big_rect.midleft, rect.center): (rect.midleft, rect.center), (big_rect.midtop, rect.center): (rect.midtop, rect.center), (big_rect.midright, rect.center): ((rect.midright[0] - 1, rect.midright[1]), rect.center), (big_rect.midbottom, rect.center): ((rect.midbottom[0], rect.midbottom[1] - 1), rect.center), (big_rect.topleft, rect.center): (rect.topleft, rect.center), (big_rect.topright, rect.center): ((rect.topright[0] - 1, rect.topright[1]), rect.center), (big_rect.bottomright, rect.center): ((rect.bottomright[0] - 1, rect.bottomright[1] - 1), rect.center), ((big_rect.bottomleft[0], big_rect.bottomleft[1] - 1), rect.center): ((rect.bottomleft[0], rect.bottomleft[1] - 1), rect.center)}
    for line, expected_line in line_dict.items():
        clipped_line = rect.clipline(line)
        self.assertTupleEqual(clipped_line, expected_line)
        expected_line = (expected_line[1], expected_line[0])
        clipped_line = rect.clipline((line[1], line[0]))
        self.assertTupleEqual(clipped_line, expected_line)