import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_ints_unequal(self):
    try:
        self.assertEqual(3, 4)
    except AssertionError as e:
        self.assertEqual(str(e), '3 != 4')