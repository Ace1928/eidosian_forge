from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_draw(self):
    """Ensures draw works for subclassed Masks."""
    mask_size = (5, 4)
    expected_count = mask_size[0] * mask_size[1]
    arg_masks = (pygame.mask.Mask(fill=True, size=mask_size), SubMask(mask_size, True))
    for mask in (pygame.mask.Mask(mask_size), SubMask(mask_size)):
        for arg_mask in arg_masks:
            mask.clear()
            mask.draw(arg_mask, (0, 0))
            self.assertEqual(mask.count(), expected_count)