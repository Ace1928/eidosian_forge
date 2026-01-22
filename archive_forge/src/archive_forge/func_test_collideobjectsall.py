import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_collideobjectsall(self):
    r = Rect(1, 1, 10, 10)
    types_to_test = [[Rect(1, 1, 10, 10), Rect(5, 5, 10, 10), Rect(15, 15, 1, 1), Rect(2, 2, 1, 1)], [(1, 1, 10, 10), (5, 5, 10, 10), (15, 15, 1, 1), (2, 2, 1, 1)], [((1, 1), (10, 10)), ((5, 5), (10, 10)), ((15, 15), (1, 1)), ((2, 2), (1, 1))], [[1, 1, 10, 10], [5, 5, 10, 10], [15, 15, 1, 1], [2, 2, 1, 1]], [self._ObjectWithRectAttribute(Rect(1, 1, 10, 10)), self._ObjectWithRectAttribute(Rect(5, 5, 10, 10)), self._ObjectWithRectAttribute(Rect(15, 15, 1, 1)), self._ObjectWithRectAttribute(Rect(2, 2, 1, 1))], [self._ObjectWithCallableRectAttribute(Rect(1, 1, 10, 10)), self._ObjectWithCallableRectAttribute(Rect(5, 5, 10, 10)), self._ObjectWithCallableRectAttribute(Rect(15, 15, 1, 1)), self._ObjectWithCallableRectAttribute(Rect(2, 2, 1, 1))], [self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(1, 1, 10, 10))), self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(5, 5, 10, 10))), self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(15, 15, 1, 1))), self._ObjectWithCallableRectAttribute(self._ObjectWithRectAttribute(Rect(2, 2, 1, 1)))], [self._ObjectWithRectProperty(Rect(1, 1, 10, 10)), self._ObjectWithRectProperty(Rect(5, 5, 10, 10)), self._ObjectWithRectProperty(Rect(15, 15, 1, 1)), self._ObjectWithRectProperty(Rect(2, 2, 1, 1))]]
    for things in types_to_test:
        with self.subTest(type=things[0].__class__.__name__):
            actual = r.collideobjectsall(things, key=None)
            self.assertEqual(actual, [things[0], things[1], things[3]])
    types_to_test = [[Rect(50, 50, 1, 1), Rect(20, 20, 5, 5)], [(50, 50, 1, 1), (20, 20, 5, 5)], [((50, 50), (1, 1)), ((20, 20), (5, 5))], [[50, 50, 1, 1], [20, 20, 5, 5]], [self._ObjectWithRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithRectAttribute(Rect(20, 20, 5, 5))], [self._ObjectWithCallableRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithCallableRectAttribute(Rect(20, 20, 5, 5))], [self._ObjectWithCallableRectAttribute(Rect(50, 50, 1, 1)), self._ObjectWithCallableRectAttribute(Rect(20, 20, 5, 5))], [self._ObjectWithRectProperty(Rect(50, 50, 1, 1)), self._ObjectWithRectProperty(Rect(20, 20, 5, 5))]]
    for f in types_to_test:
        with self.subTest(type=f[0].__class__.__name__, expected=None):
            actual = r.collideobjectsall(f)
            self.assertFalse(actual)