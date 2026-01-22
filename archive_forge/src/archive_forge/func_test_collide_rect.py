import unittest
import pygame
from pygame import sprite
def test_collide_rect(self):
    self.assertTrue(pygame.sprite.collide_rect(self.s1, self.s2))
    self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s1))
    self.s2.rect.center = self.s3.rect.center
    self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s3))
    self.assertTrue(pygame.sprite.collide_rect(self.s3, self.s2))
    self.s2.rect.inflate_ip(10, 10)
    self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s3))
    self.assertTrue(pygame.sprite.collide_rect(self.s3, self.s2))
    self.s2.rect.center = (self.s1.rect.right, self.s1.rect.bottom)
    self.assertTrue(pygame.sprite.collide_rect(self.s1, self.s2))
    self.assertTrue(pygame.sprite.collide_rect(self.s2, self.s1))
    self.assertFalse(pygame.sprite.collide_rect(self.s1, self.s3))
    self.assertFalse(pygame.sprite.collide_rect(self.s3, self.s1))