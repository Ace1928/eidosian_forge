from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__negative_sized_dest_rect(self):
    """Ensures to_surface correctly handles negative sized dest rects."""
    expected_color = pygame.Color('white')
    mask = pygame.mask.Mask((3, 5), fill=True)
    dests = (pygame.Rect((0, 0), (10, -10)), pygame.Rect((0, 0), (-10, 10)), pygame.Rect((0, 0), (-10, -10)))
    for dest in dests:
        to_surface = mask.to_surface(dest=dest)
        assertSurfaceFilled(self, to_surface, expected_color)