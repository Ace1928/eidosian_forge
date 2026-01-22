from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__different_depths(self):
    """Ensures an exception is raised when surfaces have different depths."""
    size = (13, 17)
    surface_color = pygame.Color('red')
    setsurface_color = pygame.Color('green')
    unsetsurface_color = pygame.Color('blue')
    mask = pygame.mask.Mask(size)
    test_depths = ((8, 8, 16), (8, 8, 24), (8, 8, 32), (16, 16, 24), (16, 16, 32), (24, 16, 8), (32, 16, 16), (32, 32, 16), (32, 24, 32))
    for depths in test_depths:
        surface = pygame.Surface(size, depth=depths[0])
        setsurface = pygame.Surface(size, depth=depths[1])
        unsetsurface = pygame.Surface(size, depth=depths[2])
        surface.fill(surface_color)
        setsurface.fill(setsurface_color)
        unsetsurface.fill(unsetsurface_color)
        with self.assertRaises(ValueError):
            mask.to_surface(surface, setsurface, unsetsurface)