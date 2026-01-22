import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_scatter_ellipsis_value(self):
    hv.Scatter(range(10))[..., 'y']