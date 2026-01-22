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
def test_circle_zero_radius(self):
    """Ensures zero radius circles does not draw a center pixel.

        NOTE: This is backwards incompatible behaviour with 1.9.x.
        """
    surf = pygame.Surface((200, 200))
    circle_color = pygame.Color('red')
    surf_color = pygame.Color('black')
    center = (100, 100)
    radius = 0
    width = 1
    bounding_rect = self.draw_circle(surf, circle_color, center, radius, width)
    expected_rect = create_bounding_rect(surf, surf_color, center)
    self.assertEqual(bounding_rect, expected_rect)
    self.assertEqual(bounding_rect, pygame.Rect(100, 100, 0, 0))