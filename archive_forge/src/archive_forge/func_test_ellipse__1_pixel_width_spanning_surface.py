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
def test_ellipse__1_pixel_width_spanning_surface(self):
    """Ensures an ellipse with a width of 1 is drawn correctly
        when spanning the height of the surface.

        An ellipse with a width of 1 pixel is a vertical line.
        """
    ellipse_color = pygame.Color('red')
    surface_color = pygame.Color('black')
    surf_w, surf_h = (10, 20)
    surface = pygame.Surface((surf_w, surf_h))
    rect = pygame.Rect((0, 0), (1, surf_h + 2))
    positions = ((-1, -1), (0, -1), (surf_w // 2, -1), (surf_w - 1, -1), (surf_w, -1))
    for rect_pos in positions:
        surface.fill(surface_color)
        rect.topleft = rect_pos
        self.draw_ellipse(surface, ellipse_color, rect)
        self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)