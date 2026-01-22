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
def test_blit__SRCALPHA_opaque_source(self):
    src = pygame.Surface((256, 256), SRCALPHA, 32)
    dst = src.copy()
    for i, j in test_utils.rect_area_pts(src.get_rect()):
        dst.set_at((i, j), (i, 0, 0, j))
        src.set_at((i, j), (0, i, 0, 255))
    dst.blit(src, (0, 0))
    for pt in test_utils.rect_area_pts(src.get_rect()):
        self.assertEqual(dst.get_at(pt)[1], src.get_at(pt)[1])