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
def same_size(width, height, border_width):
    """Test for ellipses with the same size as the surface."""
    surface = pygame.Surface((width, height))
    self.draw_ellipse(surface, color, (0, 0, width, height), border_width)
    borders = get_border_values(surface, width, height)
    for border in borders:
        self.assertTrue(color in border)