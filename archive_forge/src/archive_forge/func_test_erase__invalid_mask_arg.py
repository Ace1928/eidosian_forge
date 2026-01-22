from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_erase__invalid_mask_arg(self):
    """Ensure erase handles invalid mask arguments correctly."""
    size = (3, 7)
    offset = (0, 0)
    mask = pygame.mask.Mask(size)
    invalid_mask = pygame.Surface(size)
    with self.assertRaises(TypeError):
        mask.erase(invalid_mask, offset)