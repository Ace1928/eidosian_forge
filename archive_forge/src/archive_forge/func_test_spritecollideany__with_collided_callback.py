import unittest
import pygame
from pygame import sprite
def test_spritecollideany__with_collided_callback(self):
    arg_dict_a = {}
    arg_dict_b = {}
    return_container = [True]

    def collided_callback(spr_a, spr_b, arg_dict_a=arg_dict_a, arg_dict_b=arg_dict_b, return_container=return_container):
        count = arg_dict_a.get(spr_a, 0)
        arg_dict_a[spr_a] = 1 + count
        count = arg_dict_b.get(spr_b, 0)
        arg_dict_b[spr_b] = 1 + count
        return return_container[0]
    expected_sprite_choices = self.ag2.sprites()
    collided_sprite = sprite.spritecollideany(self.s1, self.ag2, collided_callback)
    self.assertIn(collided_sprite, expected_sprite_choices)
    self.assertEqual(len(arg_dict_a), 1)
    self.assertEqual(arg_dict_a[self.s1], 1)
    self.assertEqual(len(arg_dict_b), 1)
    self.assertEqual(list(arg_dict_b.values())[0], 1)
    self.assertTrue(self.s2 in arg_dict_b or self.s3 in arg_dict_b)
    arg_dict_a.clear()
    arg_dict_b.clear()
    return_container[0] = False
    collided_sprite = sprite.spritecollideany(self.s1, self.ag2, collided_callback)
    self.assertIsNone(collided_sprite)
    self.assertEqual(len(arg_dict_a), 1)
    self.assertEqual(arg_dict_a[self.s1], len(self.ag2))
    self.assertEqual(len(arg_dict_b), len(self.ag2))
    for s in self.ag2:
        self.assertEqual(arg_dict_b[s], 1)