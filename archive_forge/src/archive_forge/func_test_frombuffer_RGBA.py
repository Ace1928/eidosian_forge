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
def test_frombuffer_RGBA(self):
    rgba_buffer = bytearray([255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 10, 20, 200, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 255, 255, 255, 127, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 0, 0, 0, 79, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255, 50, 200, 20, 255])
    rgba_surf = pygame.image.frombuffer(rgba_buffer, (4, 4), 'RGBA')
    self.assertEqual(rgba_surf.get_at((0, 0)), pygame.Color(255, 10, 20, 200))
    self.assertEqual(rgba_surf.get_at((1, 1)), pygame.Color(255, 255, 255, 127))
    self.assertEqual(rgba_surf.get_at((2, 2)), pygame.Color(0, 0, 0, 79))
    self.assertEqual(rgba_surf.get_at((3, 3)), pygame.Color(50, 200, 20, 255))