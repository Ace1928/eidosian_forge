from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_erase__invalid_offset_arg(self):
    """Ensure erase handles invalid offset arguments correctly."""
    size = (7, 5)
    offset = '(0, 0)'
    mask1 = pygame.mask.Mask(size)
    mask2 = pygame.mask.Mask(size)
    with self.assertRaises(TypeError):
        mask1.erase(mask2, offset)