from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__dest_on_surface(self):
    """Ensures dest values on the surface work correctly
        when using the defaults for setcolor and unsetcolor.
        """
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    width, height = size = (5, 9)
    surface = pygame.Surface(size, SRCALPHA, 32)
    surface_color = pygame.Color('red')
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        mask_rect = mask.get_rect()
        expected_color = default_setcolor if fill else default_unsetcolor
        for dest in ((x, y) for y in range(height) for x in range(width)):
            surface.fill(surface_color)
            mask_rect.topleft = dest
            to_surface = mask.to_surface(surface, dest=dest)
            self.assertIs(to_surface, surface)
            self.assertEqual(to_surface.get_size(), size)
            assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)