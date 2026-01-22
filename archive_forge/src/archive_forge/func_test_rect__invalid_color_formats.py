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
def test_rect__invalid_color_formats(self):
    """Ensures draw rect handles invalid color formats correctly."""
    pos = (1, 1)
    surface = pygame.Surface((3, 4))
    kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 1)), 'width': 1}
    for expected_color in (2.3, self):
        kwargs['color'] = expected_color
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(**kwargs)