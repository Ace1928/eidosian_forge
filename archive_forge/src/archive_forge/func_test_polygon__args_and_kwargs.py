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
def test_polygon__args_and_kwargs(self):
    """Ensures draw polygon accepts a combination of args/kwargs"""
    surface = pygame.Surface((3, 1))
    color = (255, 255, 0, 0)
    points = ((0, 1), (1, 2), (2, 3))
    width = 0
    kwargs = {'surface': surface, 'color': color, 'points': points, 'width': width}
    for name in ('surface', 'color', 'points', 'width'):
        kwargs.pop(name)
        if 'surface' == name:
            bounds_rect = self.draw_polygon(surface, **kwargs)
        elif 'color' == name:
            bounds_rect = self.draw_polygon(surface, color, **kwargs)
        elif 'points' == name:
            bounds_rect = self.draw_polygon(surface, color, points, **kwargs)
        else:
            bounds_rect = self.draw_polygon(surface, color, points, width, **kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)