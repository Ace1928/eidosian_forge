from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__different_srcalphas_with_created_surfaces(self):
    """Ensures an exception is raised when surfaces have different SRCALPHA
        flag settings than the created surface.
        """
    size = (13, 17)
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    mask = pygame.mask.Mask(size)
    for depth in (16, 32):
        for flags in ((0, 0), (SRCALPHA, 0), (0, SRCALPHA)):
            setsurface = pygame.Surface(size, flags=flags[0], depth=depth)
            unsetsurface = pygame.Surface(size, flags=flags[1], depth=depth)
            setsurface.fill(setsurface_color)
            unsetsurface.fill(unsetsurface_color)
            with self.assertRaises(ValueError):
                mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)