from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_outline(self):
    """Ensures outline correctly handles zero sized masks."""
    expected_points = []
    for size in ((61, 0), (0, 60), (0, 0)):
        mask = pygame.mask.Mask(size)
        points = mask.outline()
        self.assertListEqual(points, expected_points, f'size={size}')