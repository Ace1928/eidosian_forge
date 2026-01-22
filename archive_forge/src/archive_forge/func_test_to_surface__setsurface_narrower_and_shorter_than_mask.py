from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__setsurface_narrower_and_shorter_than_mask(self):
    """Ensures that setsurfaces narrower and shorter than the mask work
        correctly.

        For this test the setsurface's width is less than the mask's width
        and the setsurface's height is less than the mask's height.
        """
    default_setcolor = pygame.Color('white')
    default_unsetcolor = pygame.Color('black')
    mask_size = (10, 18)
    narrow_short_size = (6, 15)
    setsurface = pygame.Surface(narrow_short_size, SRCALPHA, 32)
    setsurface_color = pygame.Color('red')
    setsurface.fill(setsurface_color)
    setsurface_rect = setsurface.get_rect()
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        to_surface = mask.to_surface(setsurface=setsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), mask_size)
        if fill:
            assertSurfaceFilled(self, to_surface, setsurface_color, setsurface_rect)
            assertSurfaceFilledIgnoreArea(self, to_surface, default_setcolor, setsurface_rect)
        else:
            assertSurfaceFilled(self, to_surface, default_unsetcolor)