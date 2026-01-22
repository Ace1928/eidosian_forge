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
def test_line__arg_invalid_types(self):
    """Ensures draw line detects invalid arg types."""
    surface = pygame.Surface((2, 2))
    color = pygame.Color('blue')
    start_pos = (0, 1)
    end_pos = (1, 2)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_line(surface, color, start_pos, end_pos, '1')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_line(surface, color, start_pos, (1, 2, 3))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_line(surface, color, (1,), end_pos)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_line(surface, 2.3, start_pos, end_pos)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_line((1, 2, 3, 4), color, start_pos, end_pos)