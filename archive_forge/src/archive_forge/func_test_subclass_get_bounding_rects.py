from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_get_bounding_rects(self):
    """Ensures get_bounding_rects works for subclassed Masks."""
    expected_bounding_rects = []
    mask = SubMask((3, 2))
    bounding_rects = mask.get_bounding_rects()
    self.assertListEqual(bounding_rects, expected_bounding_rects)