import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def testConstructionXYWidthHeight(self):
    r = Rect(1, 2, 3, 4)
    self.assertEqual(1, r.left)
    self.assertEqual(2, r.top)
    self.assertEqual(3, r.width)
    self.assertEqual(4, r.height)