from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__invalid_unsetcolor_formats(self):
    """Ensures to_surface handles invalid unsetcolor formats correctly."""
    mask = pygame.mask.Mask((5, 3))
    for unsetcolor in ('green color', '#00FF00FF0', '0x00FF00FF0', (1, 2)):
        with self.assertRaises(ValueError):
            mask.to_surface(unsetcolor=unsetcolor)
    for unsetcolor in (pygame.Surface((1, 2)), pygame.Mask((2, 1)), 1.1):
        with self.assertRaises(TypeError):
            mask.to_surface(unsetcolor=unsetcolor)