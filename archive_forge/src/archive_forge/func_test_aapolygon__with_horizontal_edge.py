import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_aapolygon__with_horizontal_edge(self):
    """Ensure aapolygon draws horizontal edges correctly.

        This test creates 2 surfaces and draws a polygon on each. The pixels
        on each surface are compared to ensure they are the same. The only
        difference between the 2 polygons is that one is drawn using
        aapolygon() and the other using multiple line() calls. They should
        produce the same final drawing.

        Related to issue #622.
        """
    bg_color = pygame.Color('white')
    line_color = pygame.Color('black')
    width, height = (11, 10)
    expected_surface = pygame.Surface((width, height), 0, 32)
    expected_surface.fill(bg_color)
    surface = pygame.Surface((width, height), 0, 32)
    surface.fill(bg_color)
    points = ((0, 0), (0, height - 1), (width - 1, height - 1), (width - 1, 0))
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        pygame.gfxdraw.line(expected_surface, x1, y1, x2, y2, line_color)
    pygame.gfxdraw.aapolygon(surface, points, line_color)
    expected_surface.lock()
    surface.lock()
    for x in range(width):
        for y in range(height):
            self.assertEqual(expected_surface.get_at((x, y)), surface.get_at((x, y)), f'pos=({x}, {y})')
    surface.unlock()
    expected_surface.unlock()