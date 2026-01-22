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
def test_blit_big_rects(self):
    """SDL2 can have more than 16 bits for x, y, width, height."""
    big_surf = pygame.Surface((100, 68000), 0, 32)
    big_surf_color = (255, 0, 0)
    big_surf.fill(big_surf_color)
    background = pygame.Surface((500, 500), 0, 32)
    background_color = (0, 255, 0)
    background.fill(background_color)
    background.blit(big_surf, (100, 100), area=(0, 16000, 100, 100))
    background.blit(big_surf, (200, 200), area=(0, 32000, 100, 100))
    background.blit(big_surf, (300, 300), area=(0, 66000, 100, 100))
    self.assertEqual(background.get_at((101, 101)), big_surf_color)
    self.assertEqual(background.get_at((201, 201)), big_surf_color)
    self.assertEqual(background.get_at((301, 301)), big_surf_color)
    self.assertEqual(background.get_at((400, 301)), background_color)
    self.assertEqual(background.get_at((400, 201)), background_color)
    self.assertEqual(background.get_at((100, 201)), background_color)
    self.assertEqual(background.get_at((99, 99)), background_color)
    self.assertEqual(background.get_at((450, 450)), background_color)