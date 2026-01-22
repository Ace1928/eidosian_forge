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
def test_ellipse__thick_line(self):
    """Ensures a thick lined ellipse is drawn correctly."""
    ellipse_color = pygame.Color('yellow')
    surface_color = pygame.Color('black')
    surface = pygame.Surface((40, 40))
    rect = pygame.Rect((0, 0), (31, 23))
    rect.center = surface.get_rect().center
    for thickness in range(1, min(*rect.size) // 2 - 2):
        surface.fill(surface_color)
        self.draw_ellipse(surface, ellipse_color, rect, thickness)
        surface.lock()
        x = rect.centerx
        y_start = rect.top
        y_end = rect.top + thickness - 1
        for y in range(y_start, y_end + 1):
            self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
        self.assertEqual(surface.get_at((x, y_start - 1)), surface_color, thickness)
        self.assertEqual(surface.get_at((x, y_end + 1)), surface_color, thickness)
        x = rect.centerx
        y_start = rect.bottom - thickness
        y_end = rect.bottom - 1
        for y in range(y_start, y_end + 1):
            self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
        self.assertEqual(surface.get_at((x, y_start - 1)), surface_color, thickness)
        self.assertEqual(surface.get_at((x, y_end + 1)), surface_color, thickness)
        x_start = rect.left
        x_end = rect.left + thickness - 1
        y = rect.centery
        for x in range(x_start, x_end + 1):
            self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
        self.assertEqual(surface.get_at((x_start - 1, y)), surface_color, thickness)
        self.assertEqual(surface.get_at((x_end + 1, y)), surface_color, thickness)
        x_start = rect.right - thickness
        x_end = rect.right - 1
        y = rect.centery
        for x in range(x_start, x_end + 1):
            self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
        self.assertEqual(surface.get_at((x_start - 1, y)), surface_color, thickness)
        self.assertEqual(surface.get_at((x_end + 1, y)), surface_color, thickness)
        surface.unlock()