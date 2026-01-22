import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_average_surfaces__subclassed_surfaces(self):
    """Ensure average_surfaces accepts subclassed surfaces."""
    expected_size = (23, 17)
    expected_flags = 0
    expected_depth = 32
    expected_color = (50, 50, 50, 255)
    surfaces = []
    for color in ((40, 60, 40), (60, 40, 60)):
        s = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
        s.fill(color)
        surfaces.append(s)
    surface = pygame.transform.average_surfaces(surfaces)
    self.assertIsInstance(surface, pygame.Surface)
    self.assertNotIsInstance(surface, test_utils.SurfaceSubclass)
    self.assertEqual(surface.get_at((0, 0)), expected_color)
    self.assertEqual(surface.get_bitsize(), expected_depth)
    self.assertEqual(surface.get_size(), expected_size)
    self.assertEqual(surface.get_flags(), expected_flags)