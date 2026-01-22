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
def test_short_line_anti_aliasing(self):
    self.surface = pygame.Surface((10, 10))
    draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
    check_points = [(i, j) for i in range(3, 8) for j in range(3, 8)]

    def check_both_directions(from_pt, to_pt, should):
        self._check_antialiasing(from_pt, to_pt, should, check_points)
    brown = (127, 127, 0)
    reddish = (191, 63, 0)
    greenish = (63, 191, 0)
    check_both_directions((4, 4), (6, 5), {(5, 4): brown, (5, 5): brown})
    check_both_directions((4, 5), (6, 4), {(5, 4): brown, (5, 5): brown})
    check_both_directions((4, 4), (5, 6), {(4, 5): brown, (5, 5): brown})
    check_both_directions((5, 4), (4, 6), {(4, 5): brown, (5, 5): brown})
    check_points = [(i, j) for i in range(2, 9) for j in range(2, 9)]
    should = {(4, 3): greenish, (5, 3): brown, (6, 3): reddish, (4, 4): reddish, (5, 4): brown, (6, 4): greenish}
    check_both_directions((3, 3), (7, 4), should)
    should = {(4, 3): reddish, (5, 3): brown, (6, 3): greenish, (4, 4): greenish, (5, 4): brown, (6, 4): reddish}
    check_both_directions((3, 4), (7, 3), should)
    should = {(4, 4): greenish, (4, 5): brown, (4, 6): reddish, (5, 4): reddish, (5, 5): brown, (5, 6): greenish}
    check_both_directions((4, 3), (5, 7), should)
    should = {(4, 4): reddish, (4, 5): brown, (4, 6): greenish, (5, 4): greenish, (5, 5): brown, (5, 6): reddish}
    check_both_directions((5, 3), (4, 7), should)