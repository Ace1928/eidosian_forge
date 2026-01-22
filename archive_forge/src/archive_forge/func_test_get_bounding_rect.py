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
def test_get_bounding_rect(self):
    surf = pygame.Surface((70, 70), SRCALPHA, 32)
    surf.fill((0, 0, 0, 0))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.width, 0)
    self.assertEqual(bound_rect.height, 0)
    surf.set_at((30, 30), (255, 255, 255, 1))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.left, 30)
    self.assertEqual(bound_rect.top, 30)
    self.assertEqual(bound_rect.width, 1)
    self.assertEqual(bound_rect.height, 1)
    surf.set_at((29, 29), (255, 255, 255, 1))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.left, 29)
    self.assertEqual(bound_rect.top, 29)
    self.assertEqual(bound_rect.width, 2)
    self.assertEqual(bound_rect.height, 2)
    surf = pygame.Surface((70, 70), 0, 24)
    surf.fill((0, 0, 0))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.width, surf.get_width())
    self.assertEqual(bound_rect.height, surf.get_height())
    surf.set_colorkey((0, 0, 0))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.width, 0)
    self.assertEqual(bound_rect.height, 0)
    surf.set_at((30, 30), (255, 255, 255))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.left, 30)
    self.assertEqual(bound_rect.top, 30)
    self.assertEqual(bound_rect.width, 1)
    self.assertEqual(bound_rect.height, 1)
    surf.set_at((60, 60), (255, 255, 255))
    bound_rect = surf.get_bounding_rect()
    self.assertEqual(bound_rect.left, 30)
    self.assertEqual(bound_rect.top, 30)
    self.assertEqual(bound_rect.width, 31)
    self.assertEqual(bound_rect.height, 31)
    pygame.display.init()
    try:
        surf = pygame.Surface((4, 1), 0, 8)
        surf.fill((255, 255, 255))
        surf.get_bounding_rect()
    finally:
        pygame.display.quit()