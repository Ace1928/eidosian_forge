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
def test_blit_blend_big_rect(self):
    """test that an oversized rect works ok."""
    color = (1, 2, 3, 255)
    area = (1, 1, 30, 30)
    s1 = pygame.Surface((4, 4), 0, 32)
    r = s1.fill(special_flags=pygame.BLEND_ADD, color=color, rect=area)
    self.assertEqual(pygame.Rect((1, 1, 3, 3)), r)
    self.assertEqual(s1.get_at((0, 0)), (0, 0, 0, 255))
    self.assertEqual(s1.get_at((1, 1)), color)
    black = pygame.Color('black')
    red = pygame.Color('red')
    self.assertNotEqual(black, red)
    surf = pygame.Surface((10, 10), 0, 32)
    surf.fill(black)
    subsurf = surf.subsurface(pygame.Rect(0, 1, 10, 8))
    self.assertEqual(surf.get_at((0, 0)), black)
    self.assertEqual(surf.get_at((0, 9)), black)
    subsurf.fill(red, (0, -1, 10, 1), pygame.BLEND_RGB_ADD)
    self.assertEqual(surf.get_at((0, 0)), black)
    self.assertEqual(surf.get_at((0, 9)), black)
    subsurf.fill(red, (0, 8, 10, 1), pygame.BLEND_RGB_ADD)
    self.assertEqual(surf.get_at((0, 0)), black)
    self.assertEqual(surf.get_at((0, 9)), black)