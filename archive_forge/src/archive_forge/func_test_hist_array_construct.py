import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_hist_array_construct(self):
    self.assertEqual(Histogram(np.column_stack((self.hxs, self.sin))), self.histogram)