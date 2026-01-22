from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_component__out_of_bounds(self):
    """Ensure connected_component() checks bounds."""
    width, height = (19, 11)
    original_size = (width, height)
    original_mask = pygame.mask.Mask(original_size, fill=True)
    original_count = original_mask.count()
    for pos in ((0, -1), (-1, 0), (0, height + 1), (width + 1, 0)):
        with self.assertRaises(IndexError):
            mask = original_mask.connected_component(pos)
        self.assertEqual(original_mask.count(), original_count)
        self.assertEqual(original_mask.get_size(), original_size)