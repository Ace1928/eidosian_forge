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
def test_palette_colorkey(self):
    """test bug discovered by robertpfeiffer
        https://github.com/pygame/pygame/issues/721
        """
    surf = pygame.image.load(example_path(os.path.join('data', 'alien2.png')))
    key = surf.get_colorkey()
    self.assertEqual(surf.get_palette()[surf.map_rgb(key)], key)