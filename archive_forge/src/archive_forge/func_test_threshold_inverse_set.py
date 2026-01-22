import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_inverse_set(self):
    """changes the pixels within the threshold, and not outside."""
    surf_size = (32, 32)
    _dest_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
    _surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
    dest_surf = _dest_surf
    surf = _surf
    search_color = (55, 55, 55, 255)
    threshold = (0, 0, 0, 0)
    set_color = (245, 245, 245, 255)
    inverse_set = 1
    original_color = (10, 10, 10, 255)
    surf.fill(original_color)
    surf.set_at((0, 0), search_color)
    surf.set_at((12, 5), search_color)
    dest_surf.fill(original_color)
    dest_surf.set_at((0, 0), search_color)
    dest_surf.set_at((12, 5), search_color)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
    num_threshold_pixels = pygame.transform.threshold(dest_surf, surf, search_color=search_color, threshold=threshold, set_color=set_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR, inverse_set=1)
    self.assertEqual(num_threshold_pixels, 2)
    self.assertEqual(dest_surf.get_at((0, 0)), set_color)
    self.assertEqual(dest_surf.get_at((12, 5)), set_color)
    self.assertEqual(dest_surf.get_at((2, 2)), original_color)