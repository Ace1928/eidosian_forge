import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass
def test_gfxdraw__subclassed_surface(self):
    """Ensure pygame.gfxdraw works on subclassed surfaces."""
    surface = SurfaceSubclass((11, 13), SRCALPHA, 32)
    surface.fill(pygame.Color('blue'))
    expected_color = pygame.Color('red')
    x, y = (1, 2)
    pygame.gfxdraw.pixel(surface, x, y, expected_color)
    self.assertEqual(surface.get_at((x, y)), expected_color)