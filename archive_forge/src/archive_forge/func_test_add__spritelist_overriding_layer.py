import unittest
import pygame
from pygame import sprite
def test_add__spritelist_overriding_layer(self):
    expected_layer = 33
    sprites = []
    sprite_and_layer_count = 10
    for i in range(sprite_and_layer_count):
        sprites.append(self.sprite())
        sprites[-1].layer = i
    self.LG.add(sprites, layer=expected_layer)
    self.assertEqual(len(self.LG._spritelist), sprite_and_layer_count)
    for i in range(sprite_and_layer_count):
        layer = self.LG.get_layer_of_sprite(sprites[i])
        self.assertEqual(layer, expected_layer)