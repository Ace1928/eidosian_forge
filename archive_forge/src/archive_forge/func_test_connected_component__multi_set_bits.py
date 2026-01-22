from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_component__multi_set_bits(self):
    """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is set with a connected component of > 1 bit.
        """
    expected_size = (113, 67)
    original_mask = pygame.mask.Mask(expected_size)
    p_width, p_height = (11, 13)
    set_pos = xset, yset = (11, 21)
    expected_offset = (xset - 1, yset - 1)
    expected_pattern = pygame.mask.Mask((p_width, p_height), fill=True)
    for y in range(3, p_height):
        for x in range(1, p_width):
            if x in [y, y - 3, p_width - 4]:
                expected_pattern.set_at((x, y), 0)
    expected_count = expected_pattern.count()
    original_mask.draw(expected_pattern, expected_offset)
    mask = original_mask.connected_component(set_pos)
    self.assertIsInstance(mask, pygame.mask.Mask)
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)
    self.assertEqual(mask.overlap_area(expected_pattern, expected_offset), expected_count)
    self.assertEqual(original_mask.count(), expected_count)
    self.assertEqual(original_mask.get_size(), expected_size)
    self.assertEqual(original_mask.overlap_area(expected_pattern, expected_offset), expected_count)