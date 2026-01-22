from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_overlap(self):
    """Ensures overlap correctly handles zero sized masks.

        Tests combinations of sized and zero sized masks.
        """
    offset = (0, 0)
    for size1, size2 in zero_size_pairs(51, 42):
        msg = f'size1={size1}, size2={size2}'
        mask1 = pygame.mask.Mask(size1, fill=True)
        mask2 = pygame.mask.Mask(size2, fill=True)
        overlap_pos = mask1.overlap(mask2, offset)
        self.assertIsNone(overlap_pos, msg)