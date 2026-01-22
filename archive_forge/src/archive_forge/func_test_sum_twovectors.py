import unittest
from kivy.vector import Vector
from operator import truediv
def test_sum_twovectors(self):
    finalVector = Vector(1, 1) + Vector(1, 1)
    self.assertEqual(finalVector.x, 2)
    self.assertEqual(finalVector.y, 2)