from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_copy__override_copy(self):
    """Ensures copy works for subclassed Masks overriding copy."""
    mask = SubMaskCopy((65, 2), fill=True)
    for i, mask_copy in enumerate((mask.copy(), copy.copy(mask))):
        self.assertIsInstance(mask_copy, pygame.mask.Mask)
        self.assertIsInstance(mask_copy, SubMaskCopy)
        self.assertIsNot(mask_copy, mask)
        assertMaskEqual(self, mask_copy, mask)
        if 1 == i:
            self.assertFalse(hasattr(mask_copy, 'test_attribute'))
        else:
            self.assertTrue(mask_copy.test_attribute)