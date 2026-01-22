import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
def test_get_arraytypes(self):
    arraytypes = pygame.sndarray.get_arraytypes()
    self.assertIn('numpy', arraytypes)
    for atype in arraytypes:
        self.assertEqual(atype, 'numpy', f'unknown array type {atype}')