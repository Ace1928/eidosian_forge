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
def not_same_size(width, height, border_width, left, top):
    """Test for ellipses that aren't the same size as the surface."""
    surface = pygame.Surface((width, height))
    self.draw_ellipse(surface, color, (left, top, width - 1, height - 1), border_width)
    borders = get_border_values(surface, width, height)
    sides_touching = [color in border for border in borders].count(True)
    self.assertEqual(sides_touching, 2)