import numpy as np
import pandas as pd
import param
from holoviews import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.element.path import BaseShape
def test_hist_curve_int_edges_construct(self):
    hist = Histogram(Curve(range(3)))
    values = hist.dimension_values(1)
    edges = hist.edges
    self.assertEqual(values, np.arange(3))
    self.assertEqual(edges, np.array([-0.5, 0.5, 1.5, 2.5]))