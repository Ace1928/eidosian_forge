from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
def test_connected_components__negative_min_with_some_bits_set(self):
    """Ensures connected_components() properly handles negative min values
        when the mask has some bits set.

        Negative and zero values for the min parameter (minimum number of bits
        per connected component) equate to setting it to one.
        """
    mask_size = (64, 12)
    mask = pygame.mask.Mask(mask_size)
    expected_comps = {}
    for corner in corners(mask):
        mask.set_at(corner)
        new_mask = pygame.mask.Mask(mask_size)
        new_mask.set_at(corner)
        expected_comps[corner] = new_mask
    center = (mask_size[0] // 2, mask_size[1] // 2)
    mask.set_at(center)
    new_mask = pygame.mask.Mask(mask_size)
    new_mask.set_at(center)
    expected_comps[center] = new_mask
    mask_count = mask.count()
    connected_comps = mask.connected_components(-3)
    self.assertEqual(len(connected_comps), len(expected_comps))
    for comp in connected_comps:
        found = False
        for pt in tuple(expected_comps.keys()):
            if comp.get_at(pt):
                found = True
                assertMaskEqual(self, comp, expected_comps[pt])
                del expected_comps[pt]
                break
        self.assertTrue(found, f'missing component for pt={pt}')
    self.assertEqual(mask.count(), mask_count)
    self.assertEqual(mask.get_size(), mask_size)