import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_average_surfaces__subclassed_destination_surface(self):
    """Ensure average_surfaces accepts a destination subclassed surface."""
    expected_size = (13, 27)
    expected_flags = 0
    expected_depth = 32
    expected_color = (15, 15, 15, 255)
    surfaces = []
    for color in ((10, 10, 20), (20, 20, 10), (30, 30, 30)):
        s = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
        s.fill(color)
        surfaces.append(s)
    expected_dest_surface = surfaces.pop()
    dest_surface = pygame.transform.average_surfaces(surfaces=surfaces, dest_surface=expected_dest_surface)
    self.assertIsInstance(dest_surface, pygame.Surface)
    self.assertIsInstance(dest_surface, test_utils.SurfaceSubclass)
    self.assertIs(dest_surface, expected_dest_surface)
    self.assertEqual(dest_surface.get_at((0, 0)), expected_color)
    self.assertEqual(dest_surface.get_bitsize(), expected_depth)
    self.assertEqual(dest_surface.get_size(), expected_size)
    self.assertEqual(dest_surface.get_flags(), expected_flags)