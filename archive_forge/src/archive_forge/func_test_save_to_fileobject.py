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
def test_save_to_fileobject(self):
    s = pygame.Surface((1, 1))
    s.fill((23, 23, 23))
    bytes_stream = io.BytesIO()
    pygame.image.save(s, bytes_stream)
    bytes_stream.seek(0)
    s2 = pygame.image.load(bytes_stream, 'tga')
    self.assertEqual(s.get_at((0, 0)), s2.get_at((0, 0)))