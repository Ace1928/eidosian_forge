from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__area_default(self):
    """Ensures the default area is correct."""
    expected_color = pygame.Color('white')
    surface_color = pygame.Color('red')
    mask_size = (3, 2)
    mask = pygame.mask.Mask(mask_size, fill=True)
    mask_rect = mask.get_rect()
    surf_size = (mask_size[0] + 2, mask_size[1] + 1)
    surface = pygame.Surface(surf_size, SRCALPHA, 32)
    surface.fill(surface_color)
    to_surface = mask.to_surface(surface, setsurface=None, unsetsurface=None, unsetcolor=None)
    self.assertIs(to_surface, surface)
    self.assertEqual(to_surface.get_size(), surf_size)
    assertSurfaceFilled(self, to_surface, expected_color, mask_rect)
    assertSurfaceFilledIgnoreArea(self, to_surface, surface_color, mask_rect)