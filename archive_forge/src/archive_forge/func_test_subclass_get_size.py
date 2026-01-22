from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_get_size(self):
    """Ensures get_size works for subclassed Masks."""
    expected_size = (2, 3)
    mask = SubMask(expected_size)
    size = mask.get_size()
    self.assertEqual(size, expected_size)