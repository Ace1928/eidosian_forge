import math
import platform
import unittest
from collections.abc import Collection, Sequence
import pygame.math
from pygame.math import Vector2, Vector3
def test_project_v3_onto_z_axis(self):
    """Project onto z-axis, e.g. get the component pointing in the z-axis direction."""
    v = Vector3(2, 3, 4)
    y_axis = Vector3(0, 0, 77)
    actual = v.project(y_axis)
    self.assertEqual(0, actual.x)
    self.assertEqual(0, actual.y)
    self.assertEqual(v.z, actual.z)