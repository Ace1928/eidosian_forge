import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_cmy__all_elements_within_limits(self):
    for c in rgba_combos_Color_generator():
        c, m, y = c.cmy
        self.assertTrue(0 <= c <= 1)
        self.assertTrue(0 <= m <= 1)
        self.assertTrue(0 <= y <= 1)