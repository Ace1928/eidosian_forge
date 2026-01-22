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
def test_arc__args_missing(self):
    """Ensures draw arc detects any missing required args."""
    surface = pygame.Surface((1, 1))
    color = pygame.Color('red')
    rect = pygame.Rect((0, 0), (2, 2))
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_arc(surface, color, rect, 0.1)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_arc(surface, color, rect)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_arc(surface, color)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_arc(surface)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_arc()