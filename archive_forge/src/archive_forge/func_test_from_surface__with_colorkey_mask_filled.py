from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_from_surface__with_colorkey_mask_filled(self):
    """Ensures from_surface creates a mask with the correct bits set
        when the surface uses a colorkey.

        The surface is filled with a color that is not the colorkey color so
        the resulting masks are expected to have all bits set.
        """
    colorkeys = ((0, 0, 0), (1, 2, 3), (10, 100, 200), (255, 255, 255))
    surface_color = (50, 100, 200)
    expected_size = (11, 7)
    expected_count = expected_size[0] * expected_size[1]
    for depth in (8, 16, 24, 32):
        msg = f'depth={depth}'
        surface = pygame.Surface(expected_size, 0, depth)
        surface.fill(surface_color)
        for colorkey in colorkeys:
            surface.set_colorkey(colorkey)
            mask = pygame.mask.from_surface(surface)
            self.assertIsInstance(mask, pygame.mask.Mask, msg)
            self.assertEqual(mask.get_size(), expected_size, msg)
            self.assertEqual(mask.count(), expected_count, msg)