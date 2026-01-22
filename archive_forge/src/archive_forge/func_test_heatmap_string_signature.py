import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_heatmap_string_signature(self):
    heatmap = HeatMap([], ['a', 'b'], 'c')
    self.assertEqual(heatmap.kdims, [Dimension('a'), Dimension('b')])
    self.assertEqual(heatmap.vdims, [Dimension('c')])