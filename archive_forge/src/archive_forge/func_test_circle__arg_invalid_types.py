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
def test_circle__arg_invalid_types(self):
    """Ensures draw circle detects invalid arg types."""
    surface = pygame.Surface((2, 2))
    color = pygame.Color('blue')
    center = (1, 1)
    radius = 1
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, center, radius, 1, 'a', 1, 1, 1)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 'b', 1, 1)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 1, 'c', 1)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 1, 1, 'd')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, center, radius, '1')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, center, '2')
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, color, (1, 2, 3), radius)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle(surface, 2.3, center, radius)
    with self.assertRaises(TypeError):
        bounds_rect = self.draw_circle((1, 2, 3, 4), color, center, radius)