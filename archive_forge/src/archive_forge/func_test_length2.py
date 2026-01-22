import unittest
from kivy.vector import Vector
from operator import truediv
def test_length2(self):
    length = Vector(10, 10).length2()
    self.assertEqual(length, 200)