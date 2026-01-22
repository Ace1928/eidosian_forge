import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_pixels3d(self):
    sources = [self._make_surface(24), self._make_surface(32)]
    for surf in sources:
        self.assertFalse(surf.get_locked())
        arr = pygame.surfarray.pixels3d(surf)
        self.assertTrue(surf.get_locked())
        self._fill_array3d(arr)
        surf.unlock()
        self.assertTrue(surf.get_locked())
        del arr
        self.assertFalse(surf.get_locked())
        self.assertEqual(surf.get_locks(), ())
        self._assert_surface(surf)
    color = (1, 2, 3, 0)
    surf = self._make_surface(32, srcalpha=True)
    arr = pygame.surfarray.pixels3d(surf)
    arr[0, 0] = color[:3]
    self.assertEqual(surf.get_at((0, 0)), color)

    def do_pixels3d(surf):
        pygame.surfarray.pixels3d(surf)
    self.assertRaises(ValueError, do_pixels3d, self._make_surface(8))
    self.assertRaises(ValueError, do_pixels3d, self._make_surface(16))