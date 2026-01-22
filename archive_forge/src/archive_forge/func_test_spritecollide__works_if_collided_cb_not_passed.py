import unittest
import pygame
from pygame import sprite
def test_spritecollide__works_if_collided_cb_not_passed(self):
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False), [self.s2])