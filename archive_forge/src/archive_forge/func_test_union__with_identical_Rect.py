import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_union__with_identical_Rect(self):
    r1 = Rect(1, 2, 3, 4)
    self.assertEqual(r1, r1.union(Rect(r1)))