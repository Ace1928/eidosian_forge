from collections import OrderedDict
import copy
import platform
import random
import unittest
import sys
import pygame
from pygame.locals import *
from pygame.math import Vector2
@unittest.skipIf(IS_PYPY, 'Segfaults on pypy')
def test_get_bounding_rects(self):
    """Ensures get_bounding_rects works correctly."""
    mask_data = []
    mask_data.append(((10, 10), (((0, 0), (1, 0), (0, 1)), ((0, 3),), ((3, 3),))))
    mask_data.append(((4, 2), (((0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (3, 1)),)))
    mask_data.append(((5, 3), (((2, 0), (1, 1), (2, 1), (3, 1), (2, 2)),)))
    mask_data.append(((5, 3), (((3, 0), (2, 1), (1, 2)),)))
    mask_data.append(((5, 2), (((3, 0), (4, 0), (0, 1), (1, 1), (2, 1), (3, 1)),)))
    mask_data.append(((5, 3), (((0, 0),), ((4, 0),), ((2, 1),), ((0, 2),), ((4, 2),))))
    for size, rect_point_tuples in mask_data:
        rects = []
        mask = pygame.Mask(size)
        for rect_points in rect_point_tuples:
            rects.append(create_bounding_rect(rect_points))
            for pt in rect_points:
                mask.set_at(pt)
        expected_rects = sorted(rects, key=tuple)
        rects = mask.get_bounding_rects()
        self.assertListEqual(sorted(mask.get_bounding_rects(), key=tuple), expected_rects, f'size={size}')