import unittest
import pygame
from pygame import sprite
def test_update_with_kwargs(self):

    class test_sprite(pygame.sprite.Sprite):
        sink = []
        sink_dict = {}

        def __init__(self, *groups):
            pygame.sprite.Sprite.__init__(self, *groups)

        def update(self, *args, **kwargs):
            self.sink += args
            self.sink_dict.update(kwargs)
    s = test_sprite()
    s.update(1, 2, 3, foo=4, bar=5)
    self.assertEqual(test_sprite.sink, [1, 2, 3])
    self.assertEqual(test_sprite.sink_dict, {'foo': 4, 'bar': 5})