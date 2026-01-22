from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_centroid__empty_mask(self):
    """Ensure an empty mask's centroid is correctly calculated."""
    expected_centroid = (0, 0)
    expected_size = (101, 103)
    mask = pygame.mask.Mask(expected_size)
    centroid = mask.centroid()
    self.assertEqual(centroid, expected_centroid)
    self.assertEqual(mask.get_size(), expected_size)