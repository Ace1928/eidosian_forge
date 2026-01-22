from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_get_size(self):
    """Ensures get_size correctly handles zero sized masks."""
    for expected_size in ((41, 0), (0, 40), (0, 0)):
        mask = pygame.mask.Mask(expected_size)
        size = mask.get_size()
        self.assertEqual(size, expected_size)