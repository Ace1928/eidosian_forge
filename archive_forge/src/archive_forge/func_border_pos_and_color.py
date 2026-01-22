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
def border_pos_and_color(surface):
    """Yields each border position and its color for a given surface.

    Clockwise from the top left corner.
    """
    width, height = surface.get_size()
    right, bottom = (width - 1, height - 1)
    for x in range(width):
        pos = (x, 0)
        yield (pos, surface.get_at(pos))
    for y in range(1, height):
        pos = (right, y)
        yield (pos, surface.get_at(pos))
    for x in range(right - 1, -1, -1):
        pos = (x, bottom)
        yield (pos, surface.get_at(pos))
    for y in range(bottom - 1, 0, -1):
        pos = (0, y)
        yield (pos, surface.get_at(pos))