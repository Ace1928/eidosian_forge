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
def test_array_interface_alpha(self):
    for shifts in [[0, 8, 16, 24], [8, 16, 24, 0], [24, 16, 8, 0], [16, 8, 0, 24]]:
        masks = [255 << s for s in shifts]
        s = pygame.Surface((4, 2), pygame.SRCALPHA, 32, masks)
        self._check_interface_rgba(s, 3)