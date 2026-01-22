from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_to_surface__surface_with_param_combinations(self):
    """Ensures to_surface works with a surface value
        and combinations of other parameters.

        This tests many different parameter combinations with full and empty
        masks.
        """
    expected_ref_count = 4
    expected_flag = SRCALPHA
    expected_depth = 32
    size = (5, 3)
    dest = (0, 0)
    surface_color = pygame.Color('red')
    setsurface_color = pygame.Color('yellow')
    unsetsurface_color = pygame.Color('blue')
    setcolor = pygame.Color('green')
    unsetcolor = pygame.Color('cyan')
    surface = pygame.Surface(size, expected_flag, expected_depth)
    setsurface = surface.copy()
    unsetsurface = surface.copy()
    setsurface.fill(setsurface_color)
    unsetsurface.fill(unsetsurface_color)
    kwargs = {'surface': surface, 'setsurface': None, 'unsetsurface': None, 'setcolor': None, 'unsetcolor': None, 'dest': None}
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
                        surface.fill(surface_color)
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
                                    expected_color = surface_color
                            elif unsetsurface_param is not None:
                                expected_color = unsetsurface_color
                            elif unsetcolor_param is not None:
                                expected_color = unsetcolor
                            else:
                                expected_color = surface_color
                            to_surface = mask.to_surface(**kwargs)
                            self.assertIs(to_surface, surface)
                            if not IS_PYPY:
                                self.assertEqual(sys.getrefcount(to_surface), expected_ref_count)
                            self.assertTrue(to_surface.get_flags() & expected_flag)
                            self.assertEqual(to_surface.get_bitsize(), expected_depth)
                            self.assertEqual(to_surface.get_size(), size)
                            assertSurfaceFilled(self, to_surface, expected_color)