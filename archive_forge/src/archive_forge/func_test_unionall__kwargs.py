import math
import unittest
from collections.abc import Collection, Sequence
import platform
import random
import unittest
from pygame import Rect, Vector2
from pygame.tests import test_utils
def test_unionall__kwargs(self):
    r1 = Rect(0, 0, 1, 1)
    r2 = Rect(-2, -2, 1, 1)
    r3 = Rect(2, 2, 1, 1)
    r4 = r1.unionall(rect=[r2, r3])
    self.assertEqual(Rect(-2, -2, 5, 5), r4)