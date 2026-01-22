from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__unsetcolor_param(self):
    """Ensures to_surface accepts a unsetcolor arg/kwarg."""
    expected_ref_count = 2
    expected_flag = SRCALPHA
    expected_depth = 32
    expected_color = pygame.Color('red')
    size = (5, 3)
    mask = pygame.mask.Mask(size)
    kwargs = {'unsetcolor': expected_color}
    for use_kwargs in (True, False):
        if use_kwargs:
            to_surface = mask.to_surface(**kwargs)
        else:
            to_surface = mask.to_surface(None, None, None, None, kwargs['unsetcolor'])
        self.assertIsInstance(to_surface, pygame.Surface)
        if not IS_PYPY:
            self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
        self.assertTrue(to_surface.get_flags() & expected_flag)
        self.assertEqual(to_surface.get_bitsize(), expected_depth)
        self.assertEqual(to_surface.get_size(), size)
        assertSurfaceFilled(self, to_surface, expected_color)