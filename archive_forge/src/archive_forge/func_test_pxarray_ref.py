import unittest
import sys
import platform
import pygame
def test_pxarray_ref(self):
    sf = pygame.Surface((5, 5))
    ar = pygame.PixelArray(sf)
    ar2 = pygame.PixelArray(sf)
    self.assertEqual(sf.get_locked(), True)
    self.assertEqual(sf.get_locks(), (ar, ar2))
    del ar
    self.assertEqual(sf.get_locked(), True)
    self.assertEqual(sf.get_locks(), (ar2,))
    ar = ar2[:]
    self.assertEqual(sf.get_locked(), True)
    self.assertEqual(sf.get_locks(), (ar2,))
    del ar
    self.assertEqual(sf.get_locked(), True)
    self.assertEqual(len(sf.get_locks()), 1)