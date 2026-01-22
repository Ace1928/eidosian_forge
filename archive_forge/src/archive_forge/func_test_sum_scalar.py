import unittest
from kivy.vector import Vector
from operator import truediv
def test_sum_scalar(self):
    with self.assertRaises(TypeError):
        Vector(1, 1) + 1