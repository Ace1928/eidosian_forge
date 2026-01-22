from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__color_alphas(self):
    """Ensures the setcolor/unsetcolor alpha values are respected."""
    size = (13, 17)
    setcolor = pygame.Color('green')
    setcolor.a = 35
    unsetcolor = pygame.Color('blue')
    unsetcolor.a = 213
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        expected_color = setcolor if fill else unsetcolor
        to_surface = mask.to_surface(setcolor=setcolor, unsetcolor=unsetcolor)
        self.assertIsInstance(to_surface, pygame.Surface)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)