import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_bars_string_signature(self):
    bars = Bars([], 'a', 'b')
    self.assertEqual(bars.kdims, [Dimension('a')])
    self.assertEqual(bars.vdims, [Dimension('b')])