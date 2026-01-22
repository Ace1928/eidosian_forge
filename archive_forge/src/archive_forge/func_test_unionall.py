import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_unionall(self):
    mr1 = self.MyRect(0, 0, 1, 1)
    self.assertTrue(mr1.an_attribute)
    mr2 = mr1.unionall([Rect(-2, -2, 1, 1), Rect(2, 2, 1, 1)])
    self.assertTrue(isinstance(mr2, self.MyRect))
    self.assertRaises(AttributeError, getattr, mr2, 'an_attribute')