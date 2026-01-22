from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_overlap_mask(self):
    """Ensures overlap_mask works for subclassed Masks."""
    expected_size = (4, 5)
    expected_count = expected_size[0] * expected_size[1]
    masks = (pygame.mask.Mask(fill=True, size=expected_size), SubMask(expected_size, True))
    arg_masks = (pygame.mask.Mask(fill=True, size=expected_size), SubMask(expected_size, True))
    for mask in masks:
        for arg_mask in arg_masks:
            overlap_mask = mask.overlap_mask(arg_mask, (0, 0))
            self.assertIsInstance(overlap_mask, pygame.mask.Mask)
            self.assertNotIsInstance(overlap_mask, SubMask)
            self.assertEqual(overlap_mask.count(), expected_count)
            self.assertEqual(overlap_mask.get_size(), expected_size)