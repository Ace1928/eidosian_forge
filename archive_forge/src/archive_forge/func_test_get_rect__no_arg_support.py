from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_get_rect__no_arg_support(self):
    """Ensures get_rect only supports kwargs."""
    mask = pygame.mask.Mask((4, 5))
    with self.assertRaises(TypeError):
        rect = mask.get_rect(3)
    with self.assertRaises(TypeError):
        rect = mask.get_rect((1, 2))