from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_clear__bit_boundaries(self):
    """Ensures masks of different sizes are cleared correctly.

        Tests masks of different sizes, including:
           -masks 31 to 33 bits wide (32 bit boundaries)
           -masks 63 to 65 bits wide (64 bit boundaries)
        """
    expected_count = 0
    for height in range(1, 4):
        for width in range(1, 66):
            mask = pygame.mask.Mask((width, height), fill=True)
            mask.clear()
            self.assertEqual(mask.count(), expected_count, f'size=({width}, {height})')