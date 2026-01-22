import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_average_color_considering_alpha_all_pixels_opaque(self):
    """ """
    s = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
    s.fill((0, 100, 200, 255))
    s.fill((10, 50, 100, 255), (0, 0, 16, 32))
    self.assertEqual(pygame.transform.average_color(s, consider_alpha=True), (5, 75, 150, 255))
    avg_color = pygame.transform.average_color(surface=s, rect=(16, 0, 16, 32), consider_alpha=True)
    self.assertEqual(avg_color, (0, 100, 200, 255))