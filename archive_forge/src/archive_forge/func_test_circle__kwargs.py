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
def test_circle__kwargs(self):
    """Ensures draw circle accepts the correct kwargs
        with and without a width and quadrant arguments.
        """
    kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'center': (2, 2), 'radius': 2, 'width': 1, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': False, 'draw_bottom_right': True}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'center': (1, 1), 'radius': 1}]
    for kwargs in kwargs_list:
        bounds_rect = self.draw_circle(**kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)