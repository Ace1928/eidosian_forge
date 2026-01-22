from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_invert__full(self):
    """Ensure a full mask can be inverted."""
    expected_count = 0
    expected_size = (43, 97)
    mask = pygame.mask.Mask(expected_size, fill=True)
    mask.invert()
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)