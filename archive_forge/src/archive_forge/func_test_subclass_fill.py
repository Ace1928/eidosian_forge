from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_fill(self):
    """Ensures fill works for subclassed Masks."""
    mask_size = (2, 4)
    expected_count = mask_size[0] * mask_size[1]
    mask = SubMask(fill=False, size=mask_size)
    mask.fill()
    self.assertEqual(mask.count(), expected_count)