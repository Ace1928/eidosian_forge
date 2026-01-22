import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
@unittest.skip('SDL silently removes invalid characters')
def test_load_bad_filename(self):
    u = 'a\x00b\x00c.png'
    self.assertRaises(pygame.error, imageext.load_extended, u)