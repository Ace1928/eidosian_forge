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
def testLoadJPG(self):
    """to see if we can load a jpg."""
    f = example_path('data/alien1.jpg')
    surf = pygame.image.load(f)
    with open(f, 'rb') as f:
        surf = pygame.image.load(f)