import unittest
import pygame
from pygame import sprite
def test_collide_circle_ratio__no_radius_and_ratio_of_two_twice(self):
    collided_func = sprite.collide_circle_ratio(2.0)
    expected_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
    collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
    self.assertListEqual(expected_sprites, collided_sprites)