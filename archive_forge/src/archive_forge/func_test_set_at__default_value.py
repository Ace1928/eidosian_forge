from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_set_at__default_value(self):
    """Ensure individual mask bits are set using the default value."""
    width, height = (3, 21)
    mask0 = pygame.mask.Mask((width, height))
    mask1 = pygame.mask.Mask((width, height), fill=True)
    mask0_expected_count = 1
    mask1_expected_count = mask1.count()
    expected_bit = 1
    pos = (width - 1, height - 1)
    mask0.set_at(pos)
    mask1.set_at(pos)
    self.assertEqual(mask0.get_at(pos), expected_bit)
    self.assertEqual(mask0.count(), mask0_expected_count)
    self.assertEqual(mask1.get_at(pos), expected_bit)
    self.assertEqual(mask1.count(), mask1_expected_count)