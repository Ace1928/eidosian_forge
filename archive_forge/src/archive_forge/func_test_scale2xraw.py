import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *
def test_scale2xraw(self):
    w, h = (32, 32)
    s = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    s.fill((0, 0, 0))
    pygame.draw.circle(s, (255, 0, 0), (w // 2, h // 2), w // 3)
    s2 = pygame.transform.scale(s, (w * 2, h * 2))
    s2_2 = pygame.transform.scale(s2, (w * 4, h * 4))
    s4 = pygame.transform.scale(s, (w * 4, h * 4))
    self.assertEqual(s2_2.get_rect().size, (128, 128))
    for pt in test_utils.rect_area_pts(s2_2.get_rect()):
        self.assertEqual(s2_2.get_at(pt), s4.get_at(pt))