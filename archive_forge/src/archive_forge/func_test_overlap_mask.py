from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap_mask(self):
    """Ensure overlap_mask's mask has correct bits set.

        Testing the different combinations of full/empty masks:
            (mask1-filled) 1 overlap_mask 1 (mask2-filled)
            (mask1-empty)  0 overlap_mask 1 (mask2-filled)
            (mask1-filled) 1 overlap_mask 0 (mask2-empty)
            (mask1-empty)  0 overlap_mask 0 (mask2-empty)
        """
    expected_size = (4, 4)
    offset = (0, 0)
    expected_default = pygame.mask.Mask(expected_size)
    expected_masks = {(True, True): pygame.mask.Mask(expected_size, fill=True)}
    for fill2 in (True, False):
        mask2 = pygame.mask.Mask(expected_size, fill=fill2)
        mask2_count = mask2.count()
        for fill1 in (True, False):
            key = (fill1, fill2)
            msg = f'key={key}'
            mask1 = pygame.mask.Mask(expected_size, fill=fill1)
            mask1_count = mask1.count()
            expected_mask = expected_masks.get(key, expected_default)
            overlap_mask = mask1.overlap_mask(other=mask2, offset=offset)
            self.assertIsInstance(overlap_mask, pygame.mask.Mask, msg)
            assertMaskEqual(self, overlap_mask, expected_mask, msg)
            self.assertEqual(mask1.count(), mask1_count, msg)
            self.assertEqual(mask2.count(), mask2_count, msg)
            self.assertEqual(mask1.get_size(), expected_size, msg)
            self.assertEqual(mask2.get_size(), expected_size, msg)