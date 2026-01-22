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
def test_overlap__bit_boundaries(self):
    """Ensures overlap handles masks of different sizes correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
    for height in range(2, 4):
        for width in range(2, 66):
            mask_size = (width, height)
            mask_count = width * height
            mask1 = pygame.mask.Mask(mask_size, fill=True)
            mask2 = pygame.mask.Mask(mask_size, fill=True)
            for offset in self.ORIGIN_OFFSETS:
                msg = f'size={mask_size}, offset={offset}'
                expected_pos = (max(offset[0], 0), max(offset[1], 0))
                overlap_pos = mask1.overlap(mask2, offset)
                self.assertEqual(overlap_pos, expected_pos, msg)
                self.assertEqual(mask1.count(), mask_count, msg)
                self.assertEqual(mask2.count(), mask_count, msg)
                self.assertEqual(mask1.get_size(), mask_size, msg)
                self.assertEqual(mask2.get_size(), mask_size, msg)