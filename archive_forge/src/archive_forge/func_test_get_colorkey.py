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
def test_get_colorkey(self):
    pygame.display.init()
    try:
        s = pygame.Surface((800, 600), 0, 32)
        self.assertIsNone(s.get_colorkey())
        s.set_colorkey(None)
        self.assertIsNone(s.get_colorkey())
        r, g, b, a = (20, 40, 60, 12)
        colorkey = pygame.Color(r, g, b)
        s.set_colorkey(colorkey)
        self.assertEqual(s.get_colorkey(), (r, g, b, 255))
        s.set_colorkey(colorkey, pygame.RLEACCEL)
        self.assertEqual(s.get_colorkey(), (r, g, b, 255))
        s.set_colorkey(pygame.Color(r + 1, g + 1, b + 1))
        self.assertNotEqual(s.get_colorkey(), (r, g, b, 255))
        s.set_colorkey(pygame.Color(r, g, b, a))
        self.assertEqual(s.get_colorkey(), (r, g, b, 255))
    finally:
        s = pygame.display.set_mode((200, 200), 0, 32)
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            s.get_colorkey()