import unittest
from kivy.vector import Vector
from operator import truediv
def test_rdiv_list(self):
    finalVector = (6.0, 6.0) / Vector(3.0, 3.0)
    self.assertEqual(finalVector.x, 2)
    self.assertEqual(finalVector.y, 2)