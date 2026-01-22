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
def test_line__gaps_with_thickness(self):
    """Ensures a thick line is drawn without any gaps."""
    expected_color = (255, 255, 255)
    thickness = 5
    for surface in self._create_surfaces():
        width = surface.get_width() - 1
        h = width // 5
        w = h * 5
        self.draw_line(surface, expected_color, (0, 5), (w, 5 + h), thickness)
        for x in range(w + 1):
            for y in range(3, 8):
                pos = (x, y + (x + 2) // 5)
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')