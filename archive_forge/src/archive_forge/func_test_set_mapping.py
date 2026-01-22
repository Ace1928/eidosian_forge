import unittest
import pygame
import pygame._sdl2.controller as controller
from pygame.tests.test_utils import prompt, question
def test_set_mapping(self):
    c = self._get_first_controller()
    if c:
        mapping = c.get_mapping()
        mapping['a'] = 'b3'
        mapping['y'] = 'b0'
        c.set_mapping(mapping)
        new_mapping = c.get_mapping()
        self.assertEqual(len(mapping), len(new_mapping))
        for i in mapping:
            if mapping[i] not in ('a', 'y'):
                self.assertEqual(mapping[i], new_mapping[i])
            elif i == 'a':
                self.assertEqual(new_mapping[i], mapping['y'])
            else:
                self.assertEqual(new_mapping[i], mapping['a'])
    else:
        self.skipTest('No controller connected')