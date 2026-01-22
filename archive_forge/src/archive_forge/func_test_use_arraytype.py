import unittest
from numpy import int8, int16, uint8, uint16, float32, array, alltrue
import pygame
import pygame.sndarray
def test_use_arraytype(self):

    def do_use_arraytype(atype):
        pygame.sndarray.use_arraytype(atype)
    pygame.sndarray.use_arraytype('numpy')
    self.assertEqual(pygame.sndarray.get_arraytype(), 'numpy')
    self.assertRaises(ValueError, do_use_arraytype, 'not an option')