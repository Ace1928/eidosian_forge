from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_zero_mask_angle(self):
    sizes = ((100, 0), (0, 100), (0, 0))
    for size in sizes:
        mask = pygame.mask.Mask(size)
        self.assertEqual(mask.angle(), 0.0)