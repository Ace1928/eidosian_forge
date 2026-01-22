from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_flip(self):
    sizes = ((100, 0), (0, 100), (0, 0))
    for size in sizes:
        mask = pygame.mask.Mask(size)
        mask.invert()
        self.assertEqual(mask.count(), 0)