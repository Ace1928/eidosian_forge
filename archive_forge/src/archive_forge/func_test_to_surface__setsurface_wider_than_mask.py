from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__setsurface_wider_than_mask(self):
    """Ensures that setsurfaces wider than the mask work correctly.

        For this test the setsurface's width is greater than the mask's width.
        """
    default_unsetcolor = pygame.Color('black')
    mask_size = (6, 15)
    wide_size = (11, 15)
    setsurface = pygame.Surface(wide_size, SRCALPHA, 32)
    setsurface_color = pygame.Color('red')
    setsurface.fill(setsurface_color)
    for fill in (True, False):
        mask = pygame.mask.Mask(mask_size, fill=fill)
        expected_color = setsurface_color if fill else default_unsetcolor
        to_surface = mask.to_surface(setsurface=setsurface)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), mask_size)
        assertSurfaceFilled(self, to_surface, expected_color)