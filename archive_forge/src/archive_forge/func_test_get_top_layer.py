import unittest
import pygame
from pygame import sprite
def test_get_top_layer(self):
    layers = [1, 5, 2, 8, 4, 5, 3, 88, 23, 0]
    for i in layers:
        self.LG.add(self.sprite(), layer=i)
    top_layer = self.LG.get_top_layer()
    self.assertEqual(top_layer, self.LG.get_top_layer())
    self.assertEqual(top_layer, max(layers))
    self.assertEqual(top_layer, max(self.LG._spritelayers.values()))
    self.assertEqual(top_layer, self.LG._spritelayers[self.LG._spritelist[-1]])