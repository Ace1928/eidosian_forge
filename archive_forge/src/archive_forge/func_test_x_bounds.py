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
def test_x_bounds(self):
    """ensures a circle is drawn properly when there is a negative x, or a big x."""
    surf = pygame.Surface((200, 200))
    bgcolor = (0, 0, 0, 255)
    surf.fill(bgcolor)
    color = (255, 0, 0, 255)
    width = 1
    radius = 10
    where = (0, 30)
    bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
    self.assertEqual(bounding_rect1, pygame.Rect(0, where[1] - radius, where[0] + radius, radius * 2))
    self.assertEqual(surf.get_at((where[0] if where[0] > 0 else 0, where[1])), color)
    self.assertEqual(surf.get_at((where[0] + radius + 1, where[1])), bgcolor)
    self.assertEqual(surf.get_at((where[0] + radius - 1, where[1])), color)
    surf.fill(bgcolor)
    where = (-1e+30, 80)
    bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
    self.assertEqual(bounding_rect1, pygame.Rect(where[0], where[1], 0, 0))
    self.assertEqual(surf.get_at((0 + radius, where[1])), bgcolor)
    surf.fill(bgcolor)
    where = (surf.get_width() + radius * 2, 80)
    bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
    self.assertEqual(bounding_rect1, pygame.Rect(where[0], where[1], 0, 0))
    self.assertEqual(surf.get_at((0, where[1])), bgcolor)
    self.assertEqual(surf.get_at((0 + radius // 2, where[1])), bgcolor)
    self.assertEqual(surf.get_at((surf.get_width() - 1, where[1])), bgcolor)
    self.assertEqual(surf.get_at((surf.get_width() - radius, where[1])), bgcolor)
    surf.fill(bgcolor)
    where = (-1, 80)
    bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
    self.assertEqual(bounding_rect1, pygame.Rect(0, where[1] - radius, where[0] + radius, radius * 2))
    self.assertEqual(surf.get_at((where[0] if where[0] > 0 else 0, where[1])), color)
    self.assertEqual(surf.get_at((where[0] + radius, where[1])), bgcolor)
    self.assertEqual(surf.get_at((where[0] + radius - 1, where[1])), color)