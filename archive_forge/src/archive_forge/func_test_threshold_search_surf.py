import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_search_surf(self):
    surf_size = (32, 32)
    surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
    search_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
    dest_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
    original_color = (10, 10, 10, 255)
    search_color = (55, 55, 55, 255)
    surf.fill(original_color)
    dest_surf.fill(original_color)
    surf.set_at((0, 0), search_color)
    surf.set_at((12, 5), search_color)
    search_surf.fill(search_color)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
    self.assertRaises(TypeError, pygame.transform.threshold, dest_surf, surf, search_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf)
    different_sized_surf = pygame.Surface((22, 33), pygame.SRCALPHA, 32)
    self.assertRaises(TypeError, pygame.transform.threshold, different_sized_surf, surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf)
    self.assertRaises(TypeError, pygame.transform.threshold, dest_surf, surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=different_sized_surf)
    num_threshold_pixels = pygame.transform.threshold(dest_surface=dest_surf, surface=surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf)
    num_pixels_within = 2
    self.assertEqual(num_threshold_pixels, num_pixels_within)
    dest_surf.fill(original_color)
    num_threshold_pixels = pygame.transform.threshold(dest_surf, surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf, inverse_set=True)
    self.assertEqual(num_threshold_pixels, 2)