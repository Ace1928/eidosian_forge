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
def testLoadBytesIO(self):
    """to see if we can load images with BytesIO."""
    files = ['data/alien1.png', 'data/alien1.jpg', 'data/alien1.gif', 'data/asprite.bmp']
    for fname in files:
        with self.subTest(fname=fname):
            with open(example_path(fname), 'rb') as f:
                img_bytes = f.read()
                img_file = io.BytesIO(img_bytes)
                image = pygame.image.load(img_file)