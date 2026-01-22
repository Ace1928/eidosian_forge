from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_mask_get_rect(self):
    """Ensures get_rect works for subclassed Masks."""
    expected_rect = pygame.Rect((0, 0), (65, 33))
    mask = SubMask(expected_rect.size, fill=True)
    rect = mask.get_rect()
    self.assertEqual(rect, expected_rect)