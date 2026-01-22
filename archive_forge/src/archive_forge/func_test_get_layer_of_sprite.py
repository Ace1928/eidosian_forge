import unittest
import pygame
from pygame import sprite
def test_get_layer_of_sprite(self):
    expected_layer = 666
    spr = self.sprite()
    self.LG.add(spr, layer=expected_layer)
    layer = self.LG.get_layer_of_sprite(spr)
    self.assertEqual(len(self.LG._spritelist), 1)
    self.assertEqual(layer, self.LG.get_layer_of_sprite(spr))
    self.assertEqual(layer, expected_layer)
    self.assertEqual(layer, self.LG._spritelayers[spr])