import array
import binascii
import io
import os
import tempfile
import unittest
import glob
import pathlib
from pygame.tests.test_utils import example_path, png, tostring
import pygame, pygame.image, pygame.pkgdata
def test_to_string__premultiplied(self):
    """test to make sure we can export a surface to a premultiplied alpha string"""

    def convertRGBAtoPremultiplied(surface_to_modify):
        for x in range(surface_to_modify.get_width()):
            for y in range(surface_to_modify.get_height()):
                color = surface_to_modify.get_at((x, y))
                premult_color = (color[0] * color[3] / 255, color[1] * color[3] / 255, color[2] * color[3] / 255, color[3])
                surface_to_modify.set_at((x, y), premult_color)
    test_surface = pygame.Surface((256, 256), pygame.SRCALPHA, 32)
    for x in range(test_surface.get_width()):
        for y in range(test_surface.get_height()):
            i = x + y * test_surface.get_width()
            test_surface.set_at((x, y), (i * 7 % 256, i * 13 % 256, i * 27 % 256, y))
    premultiplied_copy = test_surface.copy()
    convertRGBAtoPremultiplied(premultiplied_copy)
    self.assertPremultipliedAreEqual(pygame.image.tostring(test_surface, 'RGBA_PREMULT'), pygame.image.tostring(premultiplied_copy, 'RGBA'), pygame.image.tostring(test_surface, 'RGBA'))
    self.assertPremultipliedAreEqual(pygame.image.tostring(test_surface, 'ARGB_PREMULT'), pygame.image.tostring(premultiplied_copy, 'ARGB'), pygame.image.tostring(test_surface, 'ARGB'))
    no_alpha_surface = pygame.Surface((256, 256), 0, 24)
    self.assertRaises(ValueError, pygame.image.tostring, no_alpha_surface, 'RGBA_PREMULT')