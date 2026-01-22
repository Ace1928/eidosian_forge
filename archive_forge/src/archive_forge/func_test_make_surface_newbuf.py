import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_make_surface_newbuf(self):
    array = self.Array2D(range(10, 160, 10))
    surface = make_surface(array)
    self.assertCopy2D(surface, array)