import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_distance_to(self):
    diff = self.v1 - self.v2
    self.assertEqual(self.e1.distance_to(self.e2), math.sqrt(2))
    self.assertEqual(self.e1.distance_to((0, 1, 0)), math.sqrt(2))
    self.assertEqual(self.e1.distance_to([0, 1, 0]), math.sqrt(2))
    self.assertEqual(self.v1.distance_to(self.v2), math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z))
    self.assertEqual(self.v1.distance_to(self.t2), math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z))
    self.assertEqual(self.v1.distance_to(self.l2), math.sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z))
    self.assertEqual(self.v1.distance_to(self.v1), 0)
    self.assertEqual(self.v1.distance_to(self.t1), 0)
    self.assertEqual(self.v1.distance_to(self.l1), 0)
    self.assertEqual(self.v1.distance_to(self.v2), self.v2.distance_to(self.v1))
    self.assertEqual(self.v1.distance_to(self.t2), self.v2.distance_to(self.t1))
    self.assertEqual(self.v1.distance_to(self.l2), self.v2.distance_to(self.l1))