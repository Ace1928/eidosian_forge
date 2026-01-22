from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_centroid__single_row(self):
    """Ensure a mask's centroid is correctly calculated
        when setting points along a single row."""
    width, height = (5, 7)
    mask = pygame.mask.Mask((width, height))
    for y in range(height):
        mask.clear()
        for x in range(width):
            mask.set_at((x, y))
            expected_centroid = (x // 2, y)
            centroid = mask.centroid()
            self.assertEqual(centroid, expected_centroid)