import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_average_color(self):
    """ """
    for i in (24, 32):
        with self.subTest(f'Testing {i}-bit surface'):
            s = pygame.Surface((32, 32), 0, i)
            s.fill((0, 100, 200))
            s.fill((10, 50, 100), (0, 0, 16, 32))
            self.assertEqual(pygame.transform.average_color(s), (5, 75, 150, 0))
            avg_color = pygame.transform.average_color(surface=s, rect=(16, 0, 16, 32))
            self.assertEqual(avg_color, (0, 100, 200, 0))