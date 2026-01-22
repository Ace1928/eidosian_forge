from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_outline__with_arg(self):
    """Ensures outline correctly handles zero sized masks
        when using the skip pixels argument."""
    expected_points = []
    for size in ((66, 0), (0, 65), (0, 0)):
        mask = pygame.mask.Mask(size)
        points = mask.outline(10)
        self.assertListEqual(points, expected_points, f'size={size}')