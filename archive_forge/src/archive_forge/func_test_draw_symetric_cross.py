import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def test_draw_symetric_cross(self):
    """non-regression on issue #234 : x and y where handled inconsistently.

        Also, the result is/was different whether we fill or not the polygon.
        """
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, CROSS, 1)
    inside = [(x, 3) for x in range(1, 6)] + [(3, y) for y in range(1, 6)]
    for x in range(10):
        for y in range(10):
            if (x, y) in inside:
                self.assertEqual(self.surface.get_at((x, y)), RED)
            elif x in range(2, 5) and y < 7 or (y in range(2, 5) and x < 7):
                self.assertEqual(self.surface.get_at((x, y)), GREEN)
            else:
                self.assertEqual(self.surface.get_at((x, y)), RED)
    pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
    self.draw_polygon(self.surface, GREEN, CROSS, 0)
    inside = [(x, 3) for x in range(1, 6)] + [(3, y) for y in range(1, 6)]
    for x in range(10):
        for y in range(10):
            if x in range(2, 5) and y < 7 or (y in range(2, 5) and x < 7):
                self.assertEqual(self.surface.get_at((x, y)), GREEN, msg=str((x, y)))
            else:
                self.assertEqual(self.surface.get_at((x, y)), RED)