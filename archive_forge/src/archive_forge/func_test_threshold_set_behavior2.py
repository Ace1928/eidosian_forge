import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_set_behavior2(self):
    """raises an error when set_behavior=2 and set_color is not None."""
    from pygame.transform import threshold
    s1 = pygame.Surface((32, 32), SRCALPHA, 32)
    s2 = pygame.Surface((32, 32), SRCALPHA, 32)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
    self.assertRaises(TypeError, threshold, dest_surface=s2, surface=s1, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=(255, 0, 0), set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF)