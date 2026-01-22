from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_subclass_connected_components(self):
    """Ensures connected_components works for subclassed Masks."""
    expected_ccs = []
    mask = SubMask((5, 4))
    ccs = mask.connected_components()
    self.assertListEqual(ccs, expected_ccs)