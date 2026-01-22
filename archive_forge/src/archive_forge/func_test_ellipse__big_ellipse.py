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
def test_ellipse__big_ellipse(self):
    """Test for big ellipse that could overflow in algorithm"""
    width = 1025
    height = 1025
    border = 1
    x_value_test = int(0.4 * height)
    y_value_test = int(0.4 * height)
    surface = pygame.Surface((width, height))
    self.draw_ellipse(surface, (255, 0, 0), (0, 0, width, height), border)
    colored_pixels = 0
    for y in range(height):
        if surface.get_at((x_value_test, y)) == (255, 0, 0):
            colored_pixels += 1
    for x in range(width):
        if surface.get_at((x, y_value_test)) == (255, 0, 0):
            colored_pixels += 1
    self.assertEqual(colored_pixels, border * 4)