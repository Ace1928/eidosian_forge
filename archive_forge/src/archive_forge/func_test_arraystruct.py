import math
import operator
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame
from pygame.colordict import THECOLORS
@unittest.skipIf(IS_PYPY, 'PyPy has no ctypes')
def test_arraystruct(self):
    import pygame.tests.test_utils.arrinter as ai
    import ctypes as ct
    c_byte_p = ct.POINTER(ct.c_byte)
    c = pygame.Color(5, 7, 13, 23)
    flags = ai.PAI_CONTIGUOUS | ai.PAI_FORTRAN | ai.PAI_ALIGNED | ai.PAI_NOTSWAPPED
    for i in range(1, 5):
        c.set_length(i)
        inter = ai.ArrayInterface(c)
        self.assertEqual(inter.two, 2)
        self.assertEqual(inter.nd, 1)
        self.assertEqual(inter.typekind, 'u')
        self.assertEqual(inter.itemsize, 1)
        self.assertEqual(inter.flags, flags)
        self.assertEqual(inter.shape[0], i)
        self.assertEqual(inter.strides[0], 1)
        data = ct.cast(inter.data, c_byte_p)
        for j in range(i):
            self.assertEqual(data[j], c[j])