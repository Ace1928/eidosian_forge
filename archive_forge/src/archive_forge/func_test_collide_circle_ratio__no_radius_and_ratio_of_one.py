import unittest
import pygame
from pygame import sprite
def test_collide_circle_ratio__no_radius_and_ratio_of_one(self):
    self.assertEqual(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=sprite.collide_circle_ratio(1.0)), [self.s2])