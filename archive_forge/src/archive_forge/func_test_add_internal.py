import unittest
import pygame
from pygame import sprite
def test_add_internal(self):
    for g in self.groups:
        self.sprite.add_internal(g)
    for g in self.groups:
        self.assertIn(g, self.sprite.groups())