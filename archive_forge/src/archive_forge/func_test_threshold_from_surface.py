import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_from_surface(self):
    """Set similar pixels in 'dest_surf' to color in the 'surf'."""
    from pygame.transform import threshold
    surf = pygame.Surface((32, 32), SRCALPHA, 32)
    dest_surf = pygame.Surface((32, 32), SRCALPHA, 32)
    surf_color = (40, 40, 40, 255)
    dest_color = (255, 255, 255)
    surf.fill(surf_color)
    dest_surf.fill(dest_color)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
    num_threshold_pixels = threshold(dest_surface=dest_surf, surface=surf, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, inverse_set=1)
    self.assertEqual(num_threshold_pixels, dest_surf.get_height() * dest_surf.get_width())
    self.assertEqual(dest_surf.get_at((0, 0)), surf_color)