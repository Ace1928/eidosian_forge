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
def test_save_colorkey(self):
    """make sure the color key is not changed when saving."""
    s = pygame.Surface((10, 10), pygame.SRCALPHA, 32)
    s.fill((23, 23, 23))
    s.set_colorkey((0, 0, 0))
    colorkey1 = s.get_colorkey()
    p1 = s.get_at((0, 0))
    temp_filename = 'tmpimg.png'
    try:
        pygame.image.save(s, temp_filename)
        s2 = pygame.image.load(temp_filename)
    finally:
        os.remove(temp_filename)
    colorkey2 = s.get_colorkey()
    self.assertEqual(colorkey1, colorkey2)
    self.assertEqual(p1, s2.get_at((0, 0)))