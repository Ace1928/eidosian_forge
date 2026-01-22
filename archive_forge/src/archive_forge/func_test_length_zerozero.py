import unittest
from kivy.vector import Vector
from operator import truediv
def test_length_zerozero(self):
    length = Vector(0, 0).length()
    self.assertEqual(length, 0)