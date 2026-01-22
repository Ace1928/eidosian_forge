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
def test_aalines__kwargs(self):
    """Ensures draw aalines accepts the correct kwargs."""
    surface = pygame.Surface((4, 4))
    color = pygame.Color('yellow')
    points = ((0, 0), (1, 1), (2, 2))
    kwargs_list = [{'surface': surface, 'color': color, 'closed': False, 'points': points}]
    for kwargs in kwargs_list:
        bounds_rect = self.draw_aalines(**kwargs)
        self.assertIsInstance(bounds_rect, pygame.Rect)