from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_overlap_area__invalid_mask_arg(self):
    """Ensure overlap_area handles invalid mask arguments correctly."""
    size = (3, 5)
    offset = (0, 0)
    mask = pygame.mask.Mask(size)
    invalid_mask = pygame.Surface(size)
    with self.assertRaises(TypeError):
        overlap_count = mask.overlap_area(invalid_mask, offset)