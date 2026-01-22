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
def test_get_locked(self):

    def blit_locked_test(surface):
        newSurf = pygame.Surface((10, 10))
        try:
            newSurf.blit(surface, (0, 0))
        except pygame.error:
            return True
        else:
            return False
    surf = pygame.Surface((100, 100))
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf.lock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf.unlock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf = pygame.Surface((100, 100))
    surf.lock()
    surf.lock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf.unlock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf.unlock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf = pygame.Surface((100, 100))
    for i in range(1000):
        surf.lock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    for i in range(1000):
        surf.unlock()
    self.assertFalse(surf.get_locked())
    surf = pygame.Surface((100, 100))
    surf.unlock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))
    surf.unlock()
    self.assertIs(surf.get_locked(), blit_locked_test(surf))