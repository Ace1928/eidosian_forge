import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
def test_unequal_arrays(self):
    try:
        self.assertEqual(self.mat1, self.mat2)
        raise AssertionError('Array mismatch not raised')
    except AssertionError as e:
        if not str(e).startswith('Image not almost equal to 6 decimals\n'):
            raise self.failureException('Image data mismatch error not raised.')