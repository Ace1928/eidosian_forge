from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__same_srcalphas(self):
    """Ensures to_surface works correctly when the SRCALPHA flag is set or not."""
    size = (13, 17)
    surface_color = pygame.Color('red')
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    for depth in (16, 32):
        for flags in (0, SRCALPHA):
            surface = pygame.Surface(size, flags=flags, depth=depth)
            setsurface = pygame.Surface(size, flags=flags, depth=depth)
            unsetsurface = pygame.Surface(size, flags=flags, depth=depth)
            surface.fill(surface_color)
            setsurface.fill(setsurface_color)
            unsetsurface.fill(unsetsurface_color)
            for fill in (True, False):
                mask = pygame.mask.Mask(size, fill=fill)
                expected_color = setsurface_color if fill else unsetsurface_color
                to_surface = mask.to_surface(surface, setsurface, unsetsurface)
                self.assertIsInstance(to_surface, pygame.Surface)
                self.assertEqual(to_surface.get_size(), size)
                assertSurfaceFilled(self, to_surface, expected_color)
                if flags:
                    self.assertTrue(to_surface.get_flags() & flags)