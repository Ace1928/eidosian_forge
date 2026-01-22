from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__different_depths_with_created_surfaces(self):
    """Ensures an exception is raised when surfaces have different depths
        than the created surface.
        """
    size = (13, 17)
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    mask = pygame.mask.Mask(size)
    test_depths = ((8, 8), (16, 16), (24, 24), (24, 16), (32, 8), (32, 16), (32, 24), (16, 32))
    for set_depth, unset_depth in test_depths:
        setsurface = pygame.Surface(size, depth=set_depth)
        unsetsurface = pygame.Surface(size, depth=unset_depth)
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        with self.assertRaises(ValueError):
            mask.to_surface(setsurface=setsurface, unsetsurface=unsetsurface)