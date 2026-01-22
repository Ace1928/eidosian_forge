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
def test_line__kwargs_order_independent(self):
    """Ensures draw line's kwargs are not order dependent."""
    bounds_rect = self.draw_line(start_pos=(1, 2), end_pos=(2, 1), width=2, color=(10, 20, 30), surface=pygame.Surface((3, 2)))
    self.assertIsInstance(bounds_rect, pygame.Rect)