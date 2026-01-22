import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_dest_surf_not_change(self):
    """the pixels within the threshold.

        All pixels not within threshold are changed to set_color.
        So there should be none changed in this test.
        """
    w, h = size = (32, 32)
    threshold = (20, 20, 20, 20)
    original_color = (25, 25, 25, 25)
    original_dest_color = (65, 65, 65, 55)
    threshold_color = (10, 10, 10, 10)
    set_color = (255, 10, 10, 10)
    surf = pygame.Surface(size, pygame.SRCALPHA, 32)
    dest_surf = pygame.Surface(size, pygame.SRCALPHA, 32)
    search_surf = pygame.Surface(size, pygame.SRCALPHA, 32)
    surf.fill(original_color)
    search_surf.fill(threshold_color)
    dest_surf.fill(original_dest_color)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
    pixels_within_threshold = pygame.transform.threshold(dest_surface=dest_surf, surface=surf, search_color=None, threshold=threshold, set_color=set_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR, search_surf=search_surf)
    self.assertEqual(w * h, pixels_within_threshold)
    dest_rect = dest_surf.get_rect()
    dest_size = dest_rect.size
    self.assertEqual(size, dest_size)
    for pt in test_utils.rect_area_pts(dest_rect):
        self.assertNotEqual(dest_surf.get_at(pt), set_color)
        self.assertEqual(dest_surf.get_at(pt), original_dest_color)