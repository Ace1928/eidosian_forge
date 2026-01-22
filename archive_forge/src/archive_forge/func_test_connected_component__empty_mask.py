from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_component__empty_mask(self):
    """Ensure a mask's connected component is correctly calculated
        when the mask is empty.
        """
    expected_size = (37, 43)
    original_mask = pygame.mask.Mask(expected_size)
    original_count = original_mask.count()
    expected_count = 0
    mask = original_mask.connected_component()
    self.assertIsInstance(mask, pygame.mask.Mask)
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)
    self.assertEqual(original_mask.count(), original_count)
    self.assertEqual(original_mask.get_size(), expected_size)