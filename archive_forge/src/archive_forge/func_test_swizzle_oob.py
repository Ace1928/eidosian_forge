import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_swizzle_oob(self):
    """An out-of-bounds swizzle raises an AttributeError."""
    v = Vector2(7, 6)
    with self.assertRaises(AttributeError):
        v.xyz