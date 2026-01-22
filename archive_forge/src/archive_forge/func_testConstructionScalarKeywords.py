import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def testConstructionScalarKeywords(self):
    v = Vector3(x=1)
    self.assertEqual(v.x, 1.0)
    self.assertEqual(v.y, 1.0)
    self.assertEqual(v.z, 1.0)