import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_scale__vector2(self):
    s = pygame.Surface((32, 32))
    s2 = pygame.transform.scale(s, pygame.Vector2(64, 64))
    s3 = pygame.transform.smoothscale(s, pygame.Vector2(64, 64))
    self.assertEqual((64, 64), s2.get_size())
    self.assertEqual((64, 64), s3.get_size())