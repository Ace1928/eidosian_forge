import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_scale__destination(self):
    """see if the destination surface can be passed in to use."""
    s = pygame.Surface((32, 32))
    s2 = pygame.transform.scale(s, (64, 64))
    s3 = s2.copy()
    s3 = pygame.transform.scale(surface=s, size=(64, 64), dest_surface=s3)
    pygame.transform.scale(s, (64, 64), s2)
    self.assertRaises(ValueError, pygame.transform.scale, s, (33, 64), s3)
    s = pygame.Surface((32, 32))
    s2 = pygame.transform.smoothscale(s, (64, 64))
    s3 = s2.copy()
    s3 = pygame.transform.smoothscale(surface=s, size=(64, 64), dest_surface=s3)
    self.assertRaises(ValueError, pygame.transform.smoothscale, s, (33, 64), s3)