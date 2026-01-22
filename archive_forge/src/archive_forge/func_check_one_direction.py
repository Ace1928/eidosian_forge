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
def check_one_direction(from_point, to_point, should):
    self.draw_aaline(self.surface, FG_GREEN, from_point, to_point, True)
    for pt in check_points:
        color = should.get(pt, BG_RED)
        with self.subTest(from_pt=from_point, pt=pt, to=to_point):
            self.assertEqual(self.surface.get_at(pt), color)
    draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)