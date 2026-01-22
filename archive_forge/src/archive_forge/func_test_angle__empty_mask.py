from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_angle__empty_mask(self):
    """Ensure an empty mask's angle is correctly calculated."""
    expected_angle = 0.0
    expected_size = (107, 43)
    mask = pygame.mask.Mask(expected_size)
    angle = mask.angle()
    self.assertIsInstance(angle, float)
    self.assertAlmostEqual(angle, expected_angle)
    self.assertEqual(mask.get_size(), expected_size)