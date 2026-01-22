import os
import unittest
from pygame.tests import test_utils
from pygame.tests.test_utils import (
import pygame
from pygame.locals import *
from pygame.bufferproxy import BufferProxy
import platform
import gc
import weakref
import ctypes
def test_set_alpha__set_colorkey_rle(self):
    pygame.display.init()
    try:
        pygame.display.set_mode((640, 480))
        blit_to_surf = pygame.Surface((80, 71))
        blit_to_surf.fill((255, 255, 255))
        image = pygame.image.load(example_path(os.path.join('data', 'alien1.png')))
        image = image.convert()
        orig_colorkey = image.get_colorkey()
        image.set_alpha(90, RLEACCEL)
        blit_to_surf.blit(image, (0, 0))
        sample_pixel_rle = blit_to_surf.get_at((50, 50))
        self.assertEqual(image.get_colorkey(), orig_colorkey)
        image.set_colorkey(orig_colorkey, RLEACCEL)
        blit_to_surf.fill((255, 255, 255))
        blit_to_surf.blit(image, (0, 0))
        sample_pixel_no_rle = blit_to_surf.get_at((50, 50))
        self.assertAlmostEqual(sample_pixel_rle.r, sample_pixel_no_rle.r, delta=2)
        self.assertAlmostEqual(sample_pixel_rle.g, sample_pixel_no_rle.g, delta=2)
        self.assertAlmostEqual(sample_pixel_rle.b, sample_pixel_no_rle.b, delta=2)
    finally:
        pygame.display.quit()