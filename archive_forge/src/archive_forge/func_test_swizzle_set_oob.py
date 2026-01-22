import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
@unittest.skipIf(IS_PYPY, 'known pypy failure')
def test_swizzle_set_oob(self):
    """An out-of-bounds swizzle set raises an AttributeError."""
    v = Vector2(7, 6)
    with self.assertRaises(AttributeError):
        v.xz = (1, 1)