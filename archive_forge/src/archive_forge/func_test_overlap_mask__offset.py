from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap_mask__offset(self):
    """Ensure an offset overlap_mask's mask is correctly calculated."""
    mask1 = pygame.mask.Mask((65, 3), fill=True)
    mask2 = pygame.mask.Mask((66, 4), fill=True)
    mask1_count = mask1.count()
    mask2_count = mask2.count()
    mask1_size = mask1.get_size()
    mask2_size = mask2.get_size()
    expected_mask = pygame.Mask(mask1_size)
    rect1 = mask1.get_rect()
    rect2 = mask2.get_rect()
    for offset in self.ORIGIN_OFFSETS:
        msg = f'offset={offset}'
        rect2.topleft = offset
        overlap_rect = rect1.clip(rect2)
        expected_mask.clear()
        expected_mask.draw(pygame.Mask(overlap_rect.size, fill=True), overlap_rect.topleft)
        overlap_mask = mask1.overlap_mask(mask2, offset)
        self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
        assertMaskEqual(self, overlap_mask, expected_mask, msg)
        self.assertEqual(mask1.count(), mask1_count, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask1.get_size(), mask1_size, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)