import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_clipline__both_endpoints_outside(self):
    """Ensures lines that overlap the rect are clipped.

        Testing lines with both endpoints outside the rect.
        """
    rect = Rect((0, 0), (20, 20))
    big_rect = rect.inflate(2, 2)
    line_dict = {(big_rect.midleft, big_rect.midright): (rect.midleft, (rect.midright[0] - 1, rect.midright[1])), (big_rect.midtop, big_rect.midbottom): (rect.midtop, (rect.midbottom[0], rect.midbottom[1] - 1)), (big_rect.topleft, big_rect.bottomright): (rect.topleft, (rect.bottomright[0] - 1, rect.bottomright[1] - 1)), ((big_rect.topright[0] - 1, big_rect.topright[1]), (big_rect.bottomleft[0], big_rect.bottomleft[1] - 1)): ((rect.topright[0] - 1, rect.topright[1]), (rect.bottomleft[0], rect.bottomleft[1] - 1))}
    for line, expected_line in line_dict.items():
        clipped_line = rect.clipline(line)
        self.assertTupleEqual(clipped_line, expected_line)
        expected_line = (expected_line[1], expected_line[0])
        clipped_line = rect.clipline((line[1], line[0]))
        self.assertTupleEqual(clipped_line, expected_line)