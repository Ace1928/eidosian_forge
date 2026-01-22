import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_collection_abc(self):
    v = Vector3(3, 4, 5)
    self.assertTrue(isinstance(v, Collection))
    self.assertFalse(isinstance(v, Sequence))