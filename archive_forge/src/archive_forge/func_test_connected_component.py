from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_component(self):
    """Ensure a mask's connected component is correctly calculated."""
    width, height = (41, 27)
    expected_size = (width, height)
    original_mask = pygame.mask.Mask(expected_size)
    patterns = []
    offset = (0, 0)
    pattern = self._draw_component_pattern_x(original_mask, 3, offset)
    patterns.append((pattern, offset))
    size = 4
    offset = (width - size, 0)
    pattern = self._draw_component_pattern_plus(original_mask, size, offset)
    patterns.append((pattern, offset))
    offset = (width // 2, height // 2)
    pattern = self._draw_component_pattern_box(original_mask, 7, offset)
    patterns.append((pattern, offset))
    expected_pattern, expected_offset = patterns[-1]
    expected_count = expected_pattern.count()
    original_count = sum((p.count() for p, _ in patterns))
    mask = original_mask.connected_component()
    self.assertIsInstance(mask, pygame.mask.Mask)
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)
    self.assertEqual(mask.overlap_area(expected_pattern, expected_offset), expected_count)
    self.assertEqual(original_mask.count(), original_count)
    self.assertEqual(original_mask.get_size(), expected_size)
    for pattern, offset in patterns:
        self.assertEqual(original_mask.overlap_area(pattern, offset), pattern.count())