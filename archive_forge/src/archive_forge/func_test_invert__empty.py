from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_invert__empty(self):
    """Ensure an empty mask can be inverted."""
    width, height = (43, 97)
    expected_size = (width, height)
    expected_count = width * height
    mask = pygame.mask.Mask(expected_size)
    mask.invert()
    self.assertEqual(mask.count(), expected_count)
    self.assertEqual(mask.get_size(), expected_size)