from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__kwargs_create_surface(self):
    """Ensures to_surface accepts the correct kwargs
        when creating a surface.
        """
    expected_color = pygame.Color('black')
    size = (5, 3)
    mask = pygame.mask.Mask(size)
    setsurface = pygame.Surface(size, SRCALPHA, 32)
    setsurface_color = pygame.Color('red')
    setsurface.fill(setsurface_color)
    unsetsurface = setsurface.copy()
    unsetsurface.fill(expected_color)
    test_data = ((None, None), ('dest', (0, 0)), ('unsetcolor', expected_color), ('setcolor', pygame.Color('yellow')), ('unsetsurface', unsetsurface), ('setsurface', setsurface), ('surface', None))
    kwargs = dict(test_data)
    for name, _ in test_data:
        kwargs.pop(name)
        to_surface = mask.to_surface(**kwargs)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)