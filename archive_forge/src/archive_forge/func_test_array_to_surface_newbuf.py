import platform
import unittest
import pygame
from pygame.locals import *
from pygame.pixelcopy import surface_to_array, map_array, array_to_surface, make_surface
def test_array_to_surface_newbuf(self):
    array = self.Array2D(range(0, 15))
    self.assertNotEqual(array.content[0], self.surface.get_at_mapped((0, 0)))
    array_to_surface(self.surface, array)
    self.assertCopy2D(self.surface, array)