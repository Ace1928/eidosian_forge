from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_overlap_mask__bits_set(self):
    """Ensure overlap_mask's mask has correct bits set."""
    mask1 = pygame.mask.Mask((50, 50), fill=True)
    mask2 = pygame.mask.Mask((300, 10), fill=True)
    mask1_count = mask1.count()
    mask2_count = mask2.count()
    mask1_size = mask1.get_size()
    mask2_size = mask2.get_size()
    mask3 = mask1.overlap_mask(mask2, (-1, 0))
    for i in range(50):
        for j in range(10):
            self.assertEqual(mask3.get_at((i, j)), 1, f'({i}, {j})')
    for i in range(50):
        for j in range(11, 50):
            self.assertEqual(mask3.get_at((i, j)), 0, f'({i}, {j})')
    self.assertEqual(mask1.count(), mask1_count)
    self.assertEqual(mask2.count(), mask2_count)
    self.assertEqual(mask1.get_size(), mask1_size)
    self.assertEqual(mask2.get_size(), mask2_size)