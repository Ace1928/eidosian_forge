from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_copy__independent(self):
    """Ensures copy makes an independent copy of the mask."""
    mask_set_pos = (64, 1)
    mask_copy_set_pos = (64, 2)
    mask = pygame.mask.Mask((65, 3))
    mask_copies = (mask.copy(), copy.copy(mask))
    mask.set_at(mask_set_pos)
    for mask_copy in mask_copies:
        mask_copy.set_at(mask_copy_set_pos)
        self.assertIsNot(mask_copy, mask)
        self.assertNotEqual(mask_copy.get_at(mask_set_pos), mask.get_at(mask_set_pos))
        self.assertNotEqual(mask_copy.get_at(mask_copy_set_pos), mask.get_at(mask_copy_set_pos))