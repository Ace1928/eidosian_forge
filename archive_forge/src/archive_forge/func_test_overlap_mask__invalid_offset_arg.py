from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap_mask__invalid_offset_arg(self):
    """Ensure overlap_mask handles invalid offset arguments correctly."""
    size = (5, 2)
    offset = '(0, 0)'
    mask1 = pygame.mask.Mask(size)
    mask2 = pygame.mask.Mask(size)
    with self.assertRaises(TypeError):
        overlap_mask = mask1.overlap_mask(mask2, offset)