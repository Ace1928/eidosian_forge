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
def test_src_alpha_issue_1289(self):
    """blit should be white."""
    surf1 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
    surf1.fill((255, 255, 255, 100))
    surf2 = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
    self.assertEqual(surf2.get_at((0, 0)), (0, 0, 0, 0))
    surf2.blit(surf1, (0, 0))
    self.assertEqual(surf1.get_at((0, 0)), (255, 255, 255, 100))
    self.assertEqual(surf2.get_at((0, 0)), (255, 255, 255, 100))