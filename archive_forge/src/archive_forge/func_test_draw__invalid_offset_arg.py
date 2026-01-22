from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_draw__invalid_offset_arg(self):
    """Ensure draw handles invalid offset arguments correctly."""
    size = (5, 7)
    offset = '(0, 0)'
    mask1 = pygame.mask.Mask(size)
    mask2 = pygame.mask.Mask(size)
    with self.assertRaises(TypeError):
        mask1.draw(mask2, offset)