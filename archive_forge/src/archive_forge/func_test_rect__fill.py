import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
def test_rect__fill(self):
    self.surf_w, self.surf_h = self.surf_size = (320, 200)
    self.surf = pygame.Surface(self.surf_size, pygame.SRCALPHA)
    self.color = (1, 13, 24, 205)
    rect = pygame.Rect(10, 10, 25, 20)
    drawn = self.draw_rect(self.surf, self.color, rect, 0)
    self.assertEqual(drawn, rect)
    for pt in test_utils.rect_area_pts(rect):
        color_at_pt = self.surf.get_at(pt)
        self.assertEqual(color_at_pt, self.color)
    for pt in test_utils.rect_outer_bounds(rect):
        color_at_pt = self.surf.get_at(pt)
        self.assertNotEqual(color_at_pt, self.color)
    bgcolor = pygame.Color('black')
    self.surf.fill(bgcolor)
    hrect = pygame.Rect(1, 1, self.surf_w - 2, 1)
    vrect = pygame.Rect(1, 3, 1, self.surf_h - 4)
    drawn = self.draw_rect(self.surf, self.color, hrect, 0)
    self.assertEqual(drawn, hrect)
    x, y = hrect.topleft
    w, h = hrect.size
    self.assertEqual(self.surf.get_at((x - 1, y)), bgcolor)
    self.assertEqual(self.surf.get_at((x + w, y)), bgcolor)
    for i in range(x, x + w):
        self.assertEqual(self.surf.get_at((i, y)), self.color)
    drawn = self.draw_rect(self.surf, self.color, vrect, 0)
    self.assertEqual(drawn, vrect)
    x, y = vrect.topleft
    w, h = vrect.size
    self.assertEqual(self.surf.get_at((x, y - 1)), bgcolor)
    self.assertEqual(self.surf.get_at((x, y + h)), bgcolor)
    for i in range(y, y + h):
        self.assertEqual(self.surf.get_at((x, i)), self.color)