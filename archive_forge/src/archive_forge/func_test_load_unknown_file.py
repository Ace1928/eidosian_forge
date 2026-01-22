import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def test_load_unknown_file(self):
    s = 'nonexistent.png'
    self.assertRaises(FileNotFoundError, imageext.load_extended, s)