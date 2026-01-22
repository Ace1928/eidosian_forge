import unittest
from kivy.vector import Vector
from operator import truediv
def test_rdiv_scalar(self):
    finalVector = 6 / Vector(3, 3)
    self.assertEqual(finalVector.x, 2)
    self.assertEqual(finalVector.y, 2)