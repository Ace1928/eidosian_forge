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
def test_illumine_shape(self):
    """non-regression on issue #313"""
    rect = pygame.Rect((0, 0, 20, 20))
    path_data = [(0, 0), (rect.width - 1, 0), (rect.width - 5, 5 - 1), (5 - 1, 5 - 1), (5 - 1, rect.height - 5), (0, rect.height - 1)]
    pygame.draw.rect(self.surface, RED, (0, 0, 20, 20), 0)
    self.draw_polygon(self.surface, GREEN, path_data[:4], 0)
    for x in range(20):
        self.assertEqual(self.surface.get_at((x, 0)), GREEN)
    for x in range(4, rect.width - 5 + 1):
        self.assertEqual(self.surface.get_at((x, 4)), GREEN)
    pygame.draw.rect(self.surface, RED, (0, 0, 20, 20), 0)
    self.draw_polygon(self.surface, GREEN, path_data, 0)
    for x in range(4, rect.width - 5 + 1):
        self.assertEqual(self.surface.get_at((x, 4)), GREEN)