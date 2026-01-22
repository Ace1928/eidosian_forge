import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
def test_load_unicode_path_1(self):
    """non-ASCII unicode"""
    import shutil
    orig = example_path('data/alien1.png')
    temp = os.path.join(example_path('data'), '你好.png')
    shutil.copy(orig, temp)
    try:
        im = imageext.load_extended(temp)
    finally:
        os.remove(temp)