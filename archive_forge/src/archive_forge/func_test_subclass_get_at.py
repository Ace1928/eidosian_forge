from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_get_at(self):
    """Ensures get_at works for subclassed Masks."""
    expected_bit = 1
    mask = SubMask((3, 2), fill=True)
    bit = mask.get_at((0, 0))
    self.assertEqual(bit, expected_bit)