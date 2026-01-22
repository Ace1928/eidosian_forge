import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
def test_webstyle(self):
    c = pygame.Color('#CC00CC11')
    self.assertEqual(c.r, 204)
    self.assertEqual(c.g, 0)
    self.assertEqual(c.b, 204)
    self.assertEqual(c.a, 17)
    self.assertEqual(hex(c), hex(3422604305))
    c = pygame.Color('#CC00CC')
    self.assertEqual(c.r, 204)
    self.assertEqual(c.g, 0)
    self.assertEqual(c.b, 204)
    self.assertEqual(c.a, 255)
    self.assertEqual(hex(c), hex(3422604543))
    c = pygame.Color('0xCC00CC11')
    self.assertEqual(c.r, 204)
    self.assertEqual(c.g, 0)
    self.assertEqual(c.b, 204)
    self.assertEqual(c.a, 17)
    self.assertEqual(hex(c), hex(3422604305))
    c = pygame.Color('0xCC00CC')
    self.assertEqual(c.r, 204)
    self.assertEqual(c.g, 0)
    self.assertEqual(c.b, 204)
    self.assertEqual(c.a, 255)
    self.assertEqual(hex(c), hex(3422604543))
    self.assertRaises(ValueError, pygame.Color, '#cc00qq')
    self.assertRaises(ValueError, pygame.Color, '0xcc00qq')
    self.assertRaises(ValueError, pygame.Color, '09abcdef')
    self.assertRaises(ValueError, pygame.Color, '09abcde')
    self.assertRaises(ValueError, pygame.Color, 'quarky')