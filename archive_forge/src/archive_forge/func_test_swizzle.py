import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_swizzle(self):
    self.assertEqual(self.v1.yxz, (self.v1.y, self.v1.x, self.v1.z))
    self.assertEqual(self.v1.xxyyzzxyz, (self.v1.x, self.v1.x, self.v1.y, self.v1.y, self.v1.z, self.v1.z, self.v1.x, self.v1.y, self.v1.z))
    self.v1.xyz = self.t2
    self.assertEqual(self.v1, self.t2)
    self.v1.zxy = self.t2
    self.assertEqual(self.v1, (self.t2[1], self.t2[2], self.t2[0]))
    self.v1.yz = self.t2[:2]
    self.assertEqual(self.v1, (self.t2[1], self.t2[0], self.t2[1]))
    self.assertEqual(type(self.v1), Vector3)