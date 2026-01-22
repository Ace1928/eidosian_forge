import unittest
import platform
from numpy import (
import pygame
from pygame.locals import *
import pygame.surfarray
def test_pixels_alpha(self):
    palette = [(0, 0, 0, 0), (127, 127, 127, 0), (127, 127, 127, 85), (127, 127, 127, 170), (127, 127, 127, 255)]
    alphas = [0, 45, 86, 99, 180]
    surf = self._make_src_surface(32, srcalpha=True, palette=palette)
    self.assertFalse(surf.get_locked())
    arr = pygame.surfarray.pixels_alpha(surf)
    self.assertTrue(surf.get_locked())
    surf.unlock()
    self.assertTrue(surf.get_locked())
    for (x, y), i in self.test_points:
        self.assertEqual(arr[x, y], palette[i][3])
    for (x, y), i in self.test_points:
        alpha = alphas[i]
        arr[x, y] = alpha
        color = (127, 127, 127, alpha)
        self.assertEqual(surf.get_at((x, y)), color, 'posn: (%i, %i)' % (x, y))
    del arr
    self.assertFalse(surf.get_locked())
    self.assertEqual(surf.get_locks(), ())

    def do_pixels_alpha(surf):
        pygame.surfarray.pixels_alpha(surf)
    targets = [(8, False), (16, False), (16, True), (24, False), (32, False)]
    for bitsize, srcalpha in targets:
        self.assertRaises(ValueError, do_pixels_alpha, self._make_surface(bitsize, srcalpha))