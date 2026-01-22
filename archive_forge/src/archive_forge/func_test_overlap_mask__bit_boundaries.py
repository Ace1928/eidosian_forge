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
def test_overlap_mask__bit_boundaries(self):
    """Ensures overlap_mask handles masks of different sizes correctly.

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
            expected_mask = pygame.Mask(mask_size)
            rect1 = mask1.get_rect()
            rect2 = mask2.get_rect()
            for offset in self.ORIGIN_OFFSETS:
                msg = f'size={mask_size}, offset={offset}'
                rect2.topleft = offset
                overlap_rect = rect1.clip(rect2)
                expected_mask.clear()
                expected_mask.draw(pygame.Mask(overlap_rect.size, fill=True), overlap_rect.topleft)
                overlap_mask = mask1.overlap_mask(mask2, offset)
                self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
                assertMaskEqual(self, overlap_mask, expected_mask, msg)
                self.assertEqual(mask1.count(), mask_count, msg)
                self.assertEqual(mask2.count(), mask_count, msg)
                self.assertEqual(mask1.get_size(), mask_size, msg)
                self.assertEqual(mask2.get_size(), mask_size, msg)