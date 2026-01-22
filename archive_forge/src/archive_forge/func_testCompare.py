import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testCompare(self):
    int_vec = Vector3(3, -2, 13)
    flt_vec = Vector3(3.0, -2.0, 13.0)
    zero_vec = Vector3(0, 0, 0)
    self.assertEqual(int_vec == flt_vec, True)
    self.assertEqual(int_vec != flt_vec, False)
    self.assertEqual(int_vec != zero_vec, True)
    self.assertEqual(flt_vec == zero_vec, False)
    self.assertEqual(int_vec == (3, -2, 13), True)
    self.assertEqual(int_vec != (3, -2, 13), False)
    self.assertEqual(int_vec != [0, 0], True)
    self.assertEqual(int_vec == [0, 0], False)
    self.assertEqual(int_vec != 5, True)
    self.assertEqual(int_vec == 5, False)
    self.assertEqual(int_vec != [3, -2, 0, 1], True)
    self.assertEqual(int_vec == [3, -2, 0, 1], False)