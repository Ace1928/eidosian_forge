import unittest
import pygame
from pygame import sprite
def test_add__sprite_init_overiding_layer(self):
    expected_layer = 33
    spr = self.sprite()
    spr._layer = 55
    lrg2 = sprite.LayeredUpdates(spr, layer=expected_layer)
    layer = lrg2._spritelayers[spr]
    self.assertEqual(len(lrg2._spritelist), 1)
    self.assertEqual(layer, expected_layer)