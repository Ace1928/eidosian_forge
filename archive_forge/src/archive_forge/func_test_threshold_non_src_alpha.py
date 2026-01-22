import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_threshold_non_src_alpha(self):
    result = pygame.Surface((10, 10))
    s1 = pygame.Surface((10, 10))
    s2 = pygame.Surface((10, 10))
    s3 = pygame.Surface((10, 10))
    s4 = pygame.Surface((10, 10))
    x = s1.fill((0, 0, 0))
    s1.set_at((0, 0), (32, 20, 0))
    x = s2.fill((0, 20, 0))
    x = s3.fill((0, 0, 0))
    x = s4.fill((0, 0, 0))
    s2.set_at((0, 0), (33, 21, 0))
    s2.set_at((3, 0), (63, 61, 0))
    s3.set_at((0, 0), (112, 31, 0))
    s4.set_at((0, 0), (11, 31, 0))
    s4.set_at((1, 1), (12, 31, 0))
    self.assertEqual(s1.get_at((0, 0)), (32, 20, 0, 255))
    self.assertEqual(s2.get_at((0, 0)), (33, 21, 0, 255))
    self.assertEqual((0, 0), (s1.get_flags(), s2.get_flags()))
    similar_color = (255, 255, 255, 255)
    diff_color = (222, 0, 0, 255)
    threshold_color = (20, 20, 20, 255)
    THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
    num_threshold_pixels = pygame.transform.threshold(dest_surface=result, surface=s1, search_color=similar_color, threshold=threshold_color, set_color=diff_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR)
    self.assertEqual(num_threshold_pixels, 0)
    num_threshold_pixels = pygame.transform.threshold(dest_surface=result, surface=s1, search_color=(40, 40, 0), threshold=threshold_color, set_color=diff_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR)
    self.assertEqual(num_threshold_pixels, 1)
    self.assertEqual(result.get_at((0, 0)), diff_color)