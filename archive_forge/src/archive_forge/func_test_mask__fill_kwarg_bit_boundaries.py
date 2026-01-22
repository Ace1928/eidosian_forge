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
def test_mask__fill_kwarg_bit_boundaries(self):
    """Ensures masks are created correctly using the fill keyword
        over a range of sizes.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
    for height in range(1, 4):
        for width in range(1, 66):
            expected_count = width * height
            expected_size = (width, height)
            msg = f'size={expected_size}'
            mask = pygame.mask.Mask(expected_size, fill=True)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.count(), expected_count, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)