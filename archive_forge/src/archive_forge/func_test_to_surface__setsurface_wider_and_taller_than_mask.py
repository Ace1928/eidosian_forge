from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__setsurface_wider_and_taller_than_mask(self):
    """Ensures that setsurfaces wider and taller than the mask work
        correctly.

        For this test the setsurface's width is greater than the mask's width
        and the setsurface's height is greater than the mask's height.
        """
    default_unsetcolor = pygame.Color('black')
    mask_size = (6, 8)
    wide_tall_size = (11, 15)
    setsurface = pygame.Surface(wide_tall_size, SRCALPHA, 32)
    setsurface_color = pygame.Color('red')
    setsurface.fill(setsurface_color)
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        expected_color = setsurface_color if fill else default_unsetcolor
        to_surface = mask.to_surface(setsurface=setsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), mask_size)
        assertSurfaceFilled(self, to_surface, expected_color)