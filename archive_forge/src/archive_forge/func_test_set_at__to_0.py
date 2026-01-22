from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_set_at__to_0(self):
    """Ensure individual mask bits are set to 0."""
    width, height = (11, 7)
    mask0 = pygame.mask.Mask((width, height))
    mask1 = pygame.mask.Mask((width, height), fill=True)
    mask0_expected_count = 0
    mask1_expected_count = mask1.count() - 1
    expected_bit = 0
    pos = (width - 1, height - 1)
    mask0.set_at(pos, expected_bit)
    mask1.set_at(pos, expected_bit)
    self.assertEqual(mask0.get_at(pos), expected_bit)
    self.assertEqual(mask0.count(), mask0_expected_count)
    self.assertEqual(mask1.get_at(pos), expected_bit)
    self.assertEqual(mask1.count(), mask1_expected_count)