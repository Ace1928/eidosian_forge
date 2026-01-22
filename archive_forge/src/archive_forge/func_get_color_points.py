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
def get_color_points(surface, color, bounds_rect=None, match_color=True):
    """Get all the points of a given color on the surface within the given
    bounds.

    If bounds_rect is None the full surface is checked.
    If match_color is True, all points matching the color are returned,
        otherwise all points not matching the color are returned.
    """
    get_at = surface.get_at
    if bounds_rect is None:
        x_range = range(surface.get_width())
        y_range = range(surface.get_height())
    else:
        x_range = range(bounds_rect.left, bounds_rect.right)
        y_range = range(bounds_rect.top, bounds_rect.bottom)
    surface.lock()
    if match_color:
        pts = [(x, y) for x in x_range for y in y_range if get_at((x, y)) == color]
    else:
        pts = [(x, y) for x in x_range for y in y_range if get_at((x, y)) != color]
    surface.unlock()
    return pts