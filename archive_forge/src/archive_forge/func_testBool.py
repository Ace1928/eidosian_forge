import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testBool(self):
    self.assertEqual(bool(self.zeroVec), False)
    self.assertEqual(bool(self.v1), True)
    self.assertTrue(not self.zeroVec)
    self.assertTrue(self.v1)