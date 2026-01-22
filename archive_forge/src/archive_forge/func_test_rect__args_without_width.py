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
def test_rect__args_without_width(self):
    """Ensures draw rect accepts the args without a width and borders."""
    bounds_rect = self.draw_rect(pygame.Surface((3, 5)), (0, 0, 0, 255), pygame.Rect((0, 0), (1, 1)))
    self.assertIsInstance(bounds_rect, pygame.Rect)