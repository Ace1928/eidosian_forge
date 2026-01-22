import os
import sys
import unittest
from pygame.tests.test_utils import trunk_relative_path
import pygame
from pygame import scrap
@unittest.skipIf('pygame.image' not in sys.modules, 'requires pygame.image module')
def test_put__bmp_image(self):
    """Ensures put can place a BMP image into the clipboard."""
    sf = pygame.image.load(trunk_relative_path('examples/data/asprite.bmp'))
    expected_string = pygame.image.tostring(sf, 'RGBA')
    scrap.put(pygame.SCRAP_BMP, expected_string)
    self.assertEqual(scrap.get(pygame.SCRAP_BMP), expected_string)