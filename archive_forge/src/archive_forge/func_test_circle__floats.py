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
def test_circle__floats(self):
    """Ensure that floats are accepted."""
    draw.circle(surface=pygame.Surface((4, 4)), color=(255, 255, 127), center=(1.5, 1.5), radius=1.3, width=0, draw_top_right=True, draw_top_left=True, draw_bottom_left=True, draw_bottom_right=True)
    draw.circle(surface=pygame.Surface((4, 4)), color=(255, 255, 127), center=Vector2(1.5, 1.5), radius=1.3, width=0, draw_top_right=True, draw_top_left=True, draw_bottom_left=True, draw_bottom_right=True)
    draw.circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1.3, 1.3), 1.2)