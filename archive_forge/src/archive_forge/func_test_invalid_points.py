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
def test_invalid_points(self):
    self.assertRaises(TypeError, lambda: self.draw_polygon(self.surface, RED, ((0, 0), (0, 20), (20, 20), 20), 0))