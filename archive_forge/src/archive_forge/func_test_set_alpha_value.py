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
def test_set_alpha_value(self):
    """surf.set_alpha(x), where x != None, enables blending"""
    s = pygame.Surface((1, 1), SRCALPHA, 32)
    s.fill((0, 255, 0, 128))
    s.set_alpha(255)
    s2 = pygame.Surface((1, 1), SRCALPHA, 32)
    s2.fill((255, 0, 0, 255))
    s2.blit(s, (0, 0))
    self.assertGreater(s2.get_at((0, 0))[0], 0, 'the red component should be above 0')