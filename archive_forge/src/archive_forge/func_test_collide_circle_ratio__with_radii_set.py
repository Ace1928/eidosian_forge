import unittest
import pygame
from pygame import sprite
def test_collide_circle_ratio__with_radii_set(self):
    self.s1.radius = 50
    self.s2.radius = 10
    self.s3.radius = 400
    collided_func = sprite.collide_circle_ratio(0.5)
    expected_sprites = sorted(self.ag2.sprites(), key=id)
    collided_sprites = sorted(sprite.spritecollide(self.s1, self.ag2, dokill=False, collided=collided_func), key=id)
    self.assertListEqual(expected_sprites, collided_sprites)