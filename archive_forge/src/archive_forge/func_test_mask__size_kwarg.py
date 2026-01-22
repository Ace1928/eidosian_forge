from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_mask__size_kwarg(self):
    """Ensure masks are created correctly using the size keyword."""
    width, height = (73, 83)
    expected_size = (width, height)
    fill_counts = {True: width * height, False: 0}
    for fill, expected_count in fill_counts.items():
        msg = f'fill={fill}'
        mask1 = pygame.mask.Mask(fill=fill, size=expected_size)
        mask2 = pygame.mask.Mask(size=expected_size, fill=fill)
        self.assertIsInstance(mask1, pygame.mask.Mask, msg)
        self.assertIsInstance(mask2, pygame.mask.Mask, msg)
        self.assertEqual(mask1.count(), expected_count, msg)
        self.assertEqual(mask2.count(), expected_count, msg)
        self.assertEqual(mask1.get_size(), expected_size, msg)
        self.assertEqual(mask2.get_size(), expected_size, msg)