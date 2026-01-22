from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_get_at(self):
    """Ensures get_at correctly handles zero sized masks."""
    for size in ((51, 0), (0, 50), (0, 0)):
        mask = pygame.mask.Mask(size)
        with self.assertRaises(IndexError):
            value = mask.get_at((0, 0))