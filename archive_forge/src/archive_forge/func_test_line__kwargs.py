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
def test_line__kwargs(self):
    """Ensures draw line accepts the correct kwargs
        with and without a width arg.
        """
    surface = pygame.Surface((4, 4))
    color = pygame.Color('yellow')
    start_pos = (1, 1)
    end_pos = (2, 2)
    kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}]
    for kwargs in kwargs_list:
        bounds_rect = self.draw_line(**kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)