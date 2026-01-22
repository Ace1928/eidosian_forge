from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_overlap_area(self):
    """Ensures overlap_area works for subclassed Masks."""
    mask_size = (3, 2)
    expected_count = mask_size[0] * mask_size[1]
    masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
    arg_masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
    for mask in masks:
        for arg_mask in arg_masks:
            overlap_count = mask.overlap_area(arg_mask, (0, 0))
            self.assertEqual(overlap_count, expected_count)