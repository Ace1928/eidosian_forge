from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__kwargs_invalid_types(self):
    """Ensures to_surface detects invalid kwarg types."""
    size = (3, 2)
    mask = pygame.mask.Mask(size)
    valid_kwargs = {'surface': pygame.Surface(size), 'setsurface': pygame.Surface(size), 'unsetsurface': pygame.Surface(size), 'setcolor': pygame.Color('green'), 'unsetcolor': pygame.Color('green'), 'dest': (0, 0)}
    invalid_kwargs = {'surface': (1, 2, 3, 4), 'setsurface': pygame.Color('green'), 'unsetsurface': ((1, 2), (2, 1)), 'setcolor': pygame.Mask((1, 2)), 'unsetcolor': pygame.Surface((2, 2)), 'dest': (0, 0, 0)}
    kwarg_order = ('surface', 'setsurface', 'unsetsurface', 'setcolor', 'unsetcolor', 'dest')
    for kwarg in kwarg_order:
        kwargs = dict(valid_kwargs)
        kwargs[kwarg] = invalid_kwargs[kwarg]
        with self.assertRaises(TypeError):
            mask.to_surface(**kwargs)