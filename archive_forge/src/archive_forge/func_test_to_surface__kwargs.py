from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__kwargs(self):
    """Ensures to_surface accepts the correct kwargs."""
    expected_color = pygame.Color('white')
    size = (5, 3)
    mask = pygame.mask.Mask(size, fill=True)
    surface = pygame.Surface(size)
    surface_color = pygame.Color('red')
    setsurface = surface.copy()
    setsurface.fill(expected_color)
    test_data = ((None, None), ('dest', (0, 0)), ('unsetcolor', pygame.Color('yellow')), ('setcolor', expected_color), ('unsetsurface', surface.copy()), ('setsurface', setsurface), ('surface', surface))
    kwargs = dict(test_data)
    for name, _ in test_data:
        kwargs.pop(name)
        surface.fill(surface_color)
        to_surface = mask.to_surface(**kwargs)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)