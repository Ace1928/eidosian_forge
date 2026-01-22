import gc
import operator
import platform
import sys
import unittest
import weakref
from functools import reduce
from pygame.tests.test_utils import SurfaceSubclass
import pygame
def test_compare__same_colors_within_distance(self):
    """Ensures compare works correctly with same colored surfaces."""
    size = (3, 5)
    pixelarray_result_color = pygame.Color('white')
    surface_color = (127, 127, 127, 255)
    for depth in (8, 16, 24, 32):
        expected_pixelarray_surface = pygame.Surface(size, depth=depth)
        expected_pixelarray_surface.fill(pixelarray_result_color)
        surf_a = expected_pixelarray_surface.copy()
        surf_a.fill(surface_color)
        expected_surface_color = surf_a.get_at((0, 0))
        pixelarray_a = pygame.PixelArray(surf_a)
        pixelarray_b = pygame.PixelArray(surf_a.copy())
        for distance in (0.0, 0.01, 0.1, 1.0):
            pixelarray_result = pixelarray_a.compare(pixelarray_b, distance=distance)
            self.assert_surfaces_equal(pixelarray_result.surface, expected_pixelarray_surface, (depth, distance))
            self.assert_surface_filled(pixelarray_a.surface, expected_surface_color, (depth, distance))
            self.assert_surface_filled(pixelarray_b.surface, expected_surface_color, (depth, distance))
        pixelarray_a.close()
        pixelarray_b.close()
        pixelarray_result.close()