from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_draw__offset(self):
    """Ensure an offset mask can be drawn onto another mask."""
    mask1 = pygame.mask.Mask((65, 3))
    mask2 = pygame.mask.Mask((66, 4), fill=True)
    mask2_count = mask2.count()
    mask2_size = mask2.get_size()
    expected_mask = pygame.Mask(mask1.get_size())
    rect1 = mask1.get_rect()
    rect2 = mask2.get_rect()
    for offset in self.ORIGIN_OFFSETS:
        msg = f'offset={offset}'
        rect2.topleft = offset
        overlap_rect = rect1.clip(rect2)
        expected_mask.clear()
        for x in range(overlap_rect.left, overlap_rect.right):
            for y in range(overlap_rect.top, overlap_rect.bottom):
                expected_mask.set_at((x, y))
        mask1.clear()
        mask1.draw(other=mask2, offset=offset)
        assertMaskEqual(self, mask1, expected_mask, msg)
        self.assertEqual(mask2.count(), mask2_count, msg)
        self.assertEqual(mask2.get_size(), mask2_size, msg)