import numpy as np
from holoviews import Image
from holoviews.core import BoundingBox, Dimension
from holoviews.core.element import HoloMap
from holoviews.element.comparison import ComparisonTestCase
def test_element_mismatch(self):
    try:
        self.assertEqual(self.map1_1D, self.map4_1D)
        raise AssertionError('Pane mismatch in array data not raised.')
    except AssertionError as e:
        if not str(e).startswith('Image not almost equal to 6 decimals\n'):
            raise self.failureException('Image mismatch error not raised.')