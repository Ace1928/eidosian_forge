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
def test_save_unicode_path(self):
    """save unicode object with non-ASCII chars"""
    self._unicode_save('你好.bmp')