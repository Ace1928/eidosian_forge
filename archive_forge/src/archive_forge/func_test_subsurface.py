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
def test_subsurface(self):
    surf = self._make_surface(32, srcalpha=True)
    comp = surf.copy()
    comp.blit(surf, (3, 0))
    sub = surf.subsurface((3, 0, 6, 6))
    sub.blit(surf, (0, 0))
    del sub
    self._assert_same(surf, comp)

    def do_blit(d, s):
        d.blit(s, (0, 0))
    sub = surf.subsurface((1, 1, 2, 2))
    self.assertRaises(pygame.error, do_blit, surf, sub)