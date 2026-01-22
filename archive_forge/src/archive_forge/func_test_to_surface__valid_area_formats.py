from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.expectedFailure
def test_to_surface__valid_area_formats(self):
    """Ensures to_surface handles valid area formats correctly."""
    size = (3, 5)
    surface_color = pygame.Color('red')
    expected_color = pygame.Color('white')
    surface = pygame.Surface(size)
    mask = pygame.mask.Mask(size, fill=True)
    area_pos = (0, 0)
    area_size = (2, 1)
    areas = ((area_pos[0], area_pos[1], area_size[0], area_size[1]), (area_pos, area_size), (area_pos, list(area_size)), (list(area_pos), area_size), (list(area_pos), list(area_size)), [area_pos[0], area_pos[1], area_size[0], area_size[1]], [area_pos, area_size], [area_pos, list(area_size)], [list(area_pos), area_size], [list(area_pos), list(area_size)], pygame.Rect(area_pos, area_size))
    for area in areas:
        surface.fill(surface_color)
        area_rect = pygame.Rect(area)
        to_surface = mask.to_surface(surface, area=area)
        assertSurfaceFilled(self, to_surface, expected_color, area_rect)
        assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, area_rect)