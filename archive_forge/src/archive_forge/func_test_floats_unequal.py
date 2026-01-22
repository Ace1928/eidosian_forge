import numpy as np
from holoviews.core import BoundingBox
from holoviews.element.comparison import ComparisonTestCase
def test_floats_unequal(self):
    try:
        self.assertEqual(3.5, 4.5)
    except AssertionError as e:
        if not str(e).startswith('Floats not almost equal to 6 decimals'):
            raise self.failureException('Float mismatch error not raised.')