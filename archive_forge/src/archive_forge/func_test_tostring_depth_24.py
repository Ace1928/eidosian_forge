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
def test_tostring_depth_24(self):
    test_surface = pygame.Surface((64, 256), depth=24)
    for i in range(256):
        for j in range(16):
            intensity = j * 16 + 15
            test_surface.set_at((j + 0, i), (intensity, i, i, i))
            test_surface.set_at((j + 16, i), (i, intensity, i, i))
            test_surface.set_at((j + 32, i), (i, i, intensity, i))
            test_surface.set_at((j + 32, i), (i, i, i, intensity))
    fmt = 'RGB'
    fmt_buf = pygame.image.tostring(test_surface, fmt)
    test_to_from_fmt_string = pygame.image.fromstring(fmt_buf, test_surface.get_size(), fmt)
    self._assertSurfaceEqual(test_surface, test_to_from_fmt_string, f'tostring/fromstring functions are not symmetric with "{fmt}" format')