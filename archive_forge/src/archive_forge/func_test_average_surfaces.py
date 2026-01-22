import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_average_surfaces(self):
    """ """
    SIZE = 32
    s1 = pygame.Surface((SIZE, SIZE))
    s2 = pygame.Surface((SIZE, SIZE))
    s3 = pygame.Surface((SIZE, SIZE))
    s1.fill((10, 10, 70))
    s2.fill((10, 20, 70))
    s3.fill((10, 130, 10))
    surfaces = [s1, s2, s3]
    surfaces = [s1, s2]
    sr = pygame.transform.average_surfaces(surfaces)
    self.assertEqual(sr.get_at((0, 0)), (10, 15, 70, 255))
    self.assertRaises(TypeError, pygame.transform.average_surfaces, 1)
    self.assertRaises(TypeError, pygame.transform.average_surfaces, [])
    self.assertRaises(TypeError, pygame.transform.average_surfaces, [1])
    self.assertRaises(TypeError, pygame.transform.average_surfaces, [s1, 1])
    self.assertRaises(TypeError, pygame.transform.average_surfaces, [1, s1])
    self.assertRaises(TypeError, pygame.transform.average_surfaces, [s1, s2, 1])
    self.assertRaises(TypeError, pygame.transform.average_surfaces, (s for s in [s1, s2, s3]))