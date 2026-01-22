from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__unsetsurface_narrower_than_mask_and_colors_none(self):
    """Ensures that unsetsurfaces narrower than the mask work correctly
        when setcolor and unsetcolor are set to None.

        For this test the unsetsurface's width is less than the mask's width.
        """
    default_surface_color = (0, 0, 0, 0)
    mask_size = (10, 20)
    narrow_size = (6, 20)
    unsetsurface = pygame.Surface(narrow_size, SRCALPHA, 32)
    unsetsurface_color = pygame.Color('red')
    unsetsurface.fill(unsetsurface_color)
    unsetsurface_rect = unsetsurface.get_rect()
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        to_surface = mask.to_surface(unsetsurface=unsetsurface, setcolor=None, unsetcolor=None)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), mask_size)
        if fill:
            assertSurfaceFilled(self, to_surface, default_surface_color)
        else:
            assertSurfaceFilled(self, to_surface, unsetsurface_color, unsetsurface_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, default_surface_color, unsetsurface_rect)