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
def test_save_tga(self):
    s = pygame.Surface((1, 1))
    s.fill((23, 23, 23))
    with tempfile.NamedTemporaryFile(suffix='.tga', delete=False) as f:
        temp_filename = f.name
    try:
        pygame.image.save(s, temp_filename)
        s2 = pygame.image.load(temp_filename)
        self.assertEqual(s2.get_at((0, 0)), s.get_at((0, 0)))
    finally:
        os.remove(temp_filename)