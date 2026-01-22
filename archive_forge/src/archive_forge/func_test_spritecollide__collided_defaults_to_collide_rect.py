import unittest
import pygame
from pygame import sprite
def test_spritecollide__collided_defaults_to_collide_rect(self):
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_rect), [self.s2])