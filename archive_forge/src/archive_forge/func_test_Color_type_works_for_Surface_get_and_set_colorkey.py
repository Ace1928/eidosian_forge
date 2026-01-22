import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_Color_type_works_for_Surface_get_and_set_colorkey(self):
    s = pygame.Surface((32, 32))
    c = pygame.Color(33, 22, 11, 255)
    s.set_colorkey(c)
    get_r, get_g, get_b, get_a = s.get_colorkey()
    self.assertTrue(get_r == c.r)
    self.assertTrue(get_g == c.g)
    self.assertTrue(get_b == c.b)
    self.assertTrue(get_a == c.a)