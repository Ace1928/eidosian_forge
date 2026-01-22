from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__default_surface_with_param_combinations(self):
    """Ensures to_surface works with a default surface value
        and combinations of other parameters.

        This tests many different parameter combinations with full and empty
        masks.
        """
    expected_ref_count = 2
    expected_flag = SRCALPHA
    expected_depth = 32
    size = (5, 3)
    dest = (0, 0)
    default_surface_color = (0, 0, 0, 0)
    setsurface_color = pygame.Color('yellow')
    unsetsurface_color = pygame.Color('blue')
    setcolor = pygame.Color('green')
    unsetcolor = pygame.Color('cyan')
    setsurface = pygame.Surface(size, expected_flag, expected_depth)
    unsetsurface = setsurface.copy()
    setsurface.fill(setsurface_color)
    unsetsurface.fill(unsetsurface_color)
    kwargs = {'setsurface': None, 'unsetsurface': None, 'setcolor': None, 'unsetcolor': None, 'dest': None}
    for fill in (True, False):
        mask = pygame.mask.Mask(size, fill=fill)
        for setsurface_param in (setsurface, None):
            kwargs['setsurface'] = setsurface_param
            for unsetsurface_param in (unsetsurface, None):
                kwargs['unsetsurface'] = unsetsurface_param
                for setcolor_param in (setcolor, None):
                    kwargs['setcolor'] = setcolor_param
                    for unsetcolor_param in (unsetcolor, None):
                        kwargs['unsetcolor'] = unsetcolor_param
                        for dest_param in (dest, None):
                            if dest_param is None:
                                kwargs.pop('dest', None)
                            else:
                                kwargs['dest'] = dest_param
                            if fill:
                                if setsurface_param is not None:
                                    expected_color = setsurface_color
                                elif setcolor_param is not None:
                                    expected_color = setcolor
                                else:
                                    expected_color = default_surface_color
                            elif unsetsurface_param is not None:
                                expected_color = unsetsurface_color
                            elif unsetcolor_param is not None:
                                expected_color = unsetcolor
                            else:
                                expected_color = default_surface_color
                            to_surface = mask.to_surface(**kwargs)
                            self.assertIsInstance(to_surface, pygame.Surface)
                            if not IS_PYPY:
                                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
                            self.assertTrue(to_surface.get_flags() & expected_flag)
                            self.assertEqual(to_surface.get_bitsize(), expected_depth)
                            self.assertEqual(to_surface.get_size(), size)
                            assertSurfaceFilled(self, to_surface, expected_color)