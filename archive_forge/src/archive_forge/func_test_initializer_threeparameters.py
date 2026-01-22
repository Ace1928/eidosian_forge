import unittest
from kivy.vector import Vector
from operator import truediv
def test_initializer_threeparameters(self):
    with self.assertRaises(Exception):
        Vector(1, 2, 3)