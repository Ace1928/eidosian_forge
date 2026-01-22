import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_smoothscale_by(self):
    s = pygame.Surface((32, 32))
    s2 = pygame.transform.smoothscale_by(s, 2)
    self.assertEqual((64, 64), s2.get_size())
    s2 = pygame.transform.smoothscale_by(s, factor=(2.0, 1.5))
    self.assertEqual((64, 48), s2.get_size())
    dest = pygame.Surface((64, 48))
    pygame.transform.smoothscale_by(s, (2.0, 1.5), dest_surface=dest)