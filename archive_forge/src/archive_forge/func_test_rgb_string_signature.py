import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_rgb_string_signature(self):
    img = RGB(np.zeros((2, 2, 3)), ['a', 'b'], ['R', 'G', 'B'])
    self.assertEqual(img.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(img.vdims, [Dimension('R'), Dimension('G'), Dimension('B')])