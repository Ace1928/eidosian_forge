import unittest
import pygame
from pygame import sprite
def test_add__sprite_with_layer_attribute(self):
    expected_layer = 100
    spr = self.sprite()
    spr._layer = expected_layer
    self.LG.add(spr)
    layer = self.LG.get_layer_of_sprite(spr)
    self.assertEqual(len(self.LG._spritelist), 1)
    self.assertEqual(layer, expected_layer)