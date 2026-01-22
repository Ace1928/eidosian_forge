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
def test_circle__args(self):
    """Ensures draw circle accepts the correct args."""
    bounds_rect = self.draw_circle(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), 3, 1, 1, 0, 1, 1)
    self.assertIsInstance(bounds_rect, pygame.Rect)