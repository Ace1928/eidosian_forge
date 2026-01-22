import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_contours_string_signature(self):
    contours = Contours([], ['a', 'b'])
    self.assertEqual(contours.kdims, [Dimension('a'), Dimension('b')])