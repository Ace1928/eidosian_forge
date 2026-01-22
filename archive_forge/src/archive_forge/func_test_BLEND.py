import unittest
import pygame
from pygame.locals import *
def test_BLEND(self):
    """BLEND_ tests."""
    s = pygame.Surface((1, 1), SRCALPHA, 32)
    s.fill((255, 255, 255, 0))
    d = pygame.Surface((1, 1), SRCALPHA, 32)
    d.fill((0, 0, 255, 255))
    s.blit(d, (0, 0), None, BLEND_ADD)
    s.blit(d, (0, 0), None, BLEND_RGBA_ADD)
    self.assertEqual(s.get_at((0, 0))[3], 255)
    s.fill((20, 255, 255, 0))
    d.fill((10, 0, 255, 255))
    s.blit(d, (0, 0), None, BLEND_ADD)
    self.assertEqual(s.get_at((0, 0))[2], 255)
    s.fill((20, 255, 255, 0))
    d.fill((10, 0, 255, 255))
    s.blit(d, (0, 0), None, BLEND_SUB)
    self.assertEqual(s.get_at((0, 0))[0], 10)
    s.fill((20, 255, 255, 0))
    d.fill((30, 0, 255, 255))
    s.blit(d, (0, 0), None, BLEND_SUB)
    self.assertEqual(s.get_at((0, 0))[0], 0)