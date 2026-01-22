from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_get_rect(self):
    """Ensures get_rect correctly handles zero sized masks."""
    for expected_size in ((4, 0), (0, 4), (0, 0)):
        expected_rect = pygame.Rect((0, 0), expected_size)
        mask = pygame.mask.Mask(expected_size)
        rect = mask.get_rect()
        self.assertEqual(rect, expected_rect)