import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v2_onto_x_axis(self):
    """Project onto x-axis, e.g. get the component pointing in the x-axis direction."""
    v = Vector2(2, 2)
    x_axis = Vector2(10, 0)
    actual = v.project(x_axis)
    self.assertEqual(v.x, actual.x)
    self.assertEqual(0, actual.y)