from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_angle(self):
    """Ensures angle works for subclassed Masks."""
    expected_angle = 0.0
    mask = SubMask(size=(5, 4))
    angle = mask.angle()
    self.assertAlmostEqual(angle, expected_angle)