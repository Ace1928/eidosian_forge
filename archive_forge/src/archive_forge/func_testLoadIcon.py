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
def testLoadIcon(self):
    """see if we can load the pygame icon."""
    f = pygame.pkgdata.getResource('pygame_icon.bmp')
    self.assertEqual(f.mode, 'rb')
    surf = pygame.image.load_basic(f)
    self.assertEqual(surf.get_at((0, 0)), (5, 4, 5, 255))
    self.assertEqual(surf.get_height(), 32)
    self.assertEqual(surf.get_width(), 32)