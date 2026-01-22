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
def test_load_jpg_threads(self):
    self.threads_load(glob.glob(example_path('data/*.jpg')))