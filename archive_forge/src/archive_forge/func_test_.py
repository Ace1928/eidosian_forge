import unittest
from kivy.vector import Vector
from operator import truediv
def test_(self):
    a = (98, 28)
    b = (72, 33)
    c = (10, -5)
    d = (20, 88)
    result = Vector.line_intersection(a, b, c, d)
    self.assertAlmostEqual(result.x, 15.25931928687196)
    self.assertAlmostEqual(result.y, 43.91166936790924)