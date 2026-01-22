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
def test_blend(self):
    bitsizes = [8, 16, 24, 32]
    blends = ['BLEND_ADD', 'BLEND_SUB', 'BLEND_MULT', 'BLEND_MIN', 'BLEND_MAX']
    for bitsize in bitsizes:
        surf = self._make_surface(bitsize)
        comp = self._make_surface(bitsize)
        for blend in blends:
            self._fill_surface(surf)
            self._fill_surface(comp)
            comp.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
            surf.blit(surf, (3, 0), special_flags=getattr(pygame, blend))
            self._assert_same(surf, comp)