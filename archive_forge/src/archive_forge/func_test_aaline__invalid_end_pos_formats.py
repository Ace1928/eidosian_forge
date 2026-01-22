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
def test_aaline__invalid_end_pos_formats(self):
    """Ensures draw aaline handles invalid end_pos formats correctly."""
    kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': (2, 2), 'end_pos': None}
    end_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
    for end_pos in end_pos_fmts:
        kwargs['end_pos'] = end_pos
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(**kwargs)