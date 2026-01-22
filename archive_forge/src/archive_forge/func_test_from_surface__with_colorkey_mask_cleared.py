from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_surface__with_colorkey_mask_cleared(self):
    """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with the colorkey color so the resulting masks
        are expected to have no bits set.
        """
    colorkeys = ((0, 0, 0), (1, 2, 3), (50, 100, 200), (255, 255, 255))
    expected_size = (7, 11)
    expected_count = 0
    for depth in (8, 16, 24, 32):
        msg = f'depth={depth}'
        surface = pygame.Surface(expected_size, 0, depth)
        for colorkey in colorkeys:
            surface.set_colorkey(colorkey)
            surface.fill(surface.get_colorkey())
            mask = pygame.mask.from_surface(surface)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)