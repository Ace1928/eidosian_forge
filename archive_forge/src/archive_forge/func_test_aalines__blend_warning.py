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
def test_aalines__blend_warning(self):
    """From pygame 2, blend=False should raise DeprecationWarning."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        self.draw_aalines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)), False)
        self.assertEqual(len(w), 1)
        self.assertTrue(issubclass(w[-1].category, DeprecationWarning))