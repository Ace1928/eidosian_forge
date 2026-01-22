import unittest
import pygame
from pygame.locals import *
def test_blits_bad_surf_args(self):
    dst = pygame.Surface((100, 10), SRCALPHA, 32)
    self.assertRaises(TypeError, dst.blits, [(None, None)])