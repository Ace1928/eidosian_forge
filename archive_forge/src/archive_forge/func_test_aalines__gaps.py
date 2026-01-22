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
def test_aalines__gaps(self):
    """Tests if the aalines drawn contain any gaps.

        Draws aalines around the border of the given surface and checks if
        all borders of the surface contain any gaps.

        See: #512
        """
    expected_color = (255, 255, 255)
    for surface in self._create_surfaces():
        self.draw_aalines(surface, expected_color, True, corners(surface))
        for pos, color in border_pos_and_color(surface):
            self.assertEqual(color, expected_color, f'pos={pos}')