from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_component__unset_bit(self):
    """Ensure a mask's connected component is correctly calculated
        when the coordinate's bit is unset.
        """
    width, height = (109, 101)
    expected_size = (width, height)
    original_mask = pygame.mask.Mask(expected_size, fill=True)
    unset_pos = (width // 2, height // 2)
    original_mask.set_at(unset_pos, 0)
    original_count = original_mask.count()
    expected_count = 0
    mask = original_mask.connected_component(unset_pos)
    self.assertIsInstance(mask, pygame.mask.Mask)
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)
    self.assertEqual(original_mask.count(), original_count)
    self.assertEqual(original_mask.get_size(), expected_size)
    self.assertEqual(original_mask.get_at(unset_pos), 0)