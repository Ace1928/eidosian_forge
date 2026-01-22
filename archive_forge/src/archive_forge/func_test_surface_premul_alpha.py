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
def test_surface_premul_alpha(self):
    """Ensure that .premul_alpha() works correctly"""
    s1 = pygame.Surface((100, 100), pygame.SRCALPHA, 32)
    s1.fill(pygame.Color(255, 255, 255, 100))
    s1_alpha = s1.premul_alpha()
    self.assertEqual(s1_alpha.get_at((50, 50)), pygame.Color(100, 100, 100, 100))
    s2 = pygame.Surface((100, 100), pygame.SRCALPHA, 16)
    s2.fill(pygame.Color(int(15 / 15 * 255), int(15 / 15 * 255), int(15 / 15 * 255), int(10 / 15 * 255)))
    s2_alpha = s2.premul_alpha()
    self.assertEqual(s2_alpha.get_at((50, 50)), pygame.Color(int(10 / 15 * 255), int(10 / 15 * 255), int(10 / 15 * 255), int(10 / 15 * 255)))
    invalid_surf = pygame.Surface((100, 100), 0, 32)
    invalid_surf.fill(pygame.Color(255, 255, 255, 100))
    with self.assertRaises(ValueError):
        invalid_surf.premul_alpha()
    test_colors = [(200, 30, 74), (76, 83, 24), (184, 21, 6), (74, 4, 74), (76, 83, 24), (184, 21, 234), (160, 30, 74), (96, 147, 204), (198, 201, 60), (132, 89, 74), (245, 9, 224), (184, 112, 6)]
    for r, g, b in test_colors:
        for a in range(255):
            with self.subTest(r=r, g=g, b=b, a=a):
                surf = pygame.Surface((10, 10), pygame.SRCALPHA, 32)
                surf.fill(pygame.Color(r, g, b, a))
                surf = surf.premul_alpha()
                self.assertEqual(surf.get_at((5, 5)), Color((r + 1) * a >> 8, (g + 1) * a >> 8, (b + 1) * a >> 8, a))