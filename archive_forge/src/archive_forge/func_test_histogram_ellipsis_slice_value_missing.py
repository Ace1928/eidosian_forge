import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_histogram_ellipsis_slice_value_missing(self):
    frequencies, edges = np.histogram(range(20), 20)
    with self.assertRaises(IndexError):
        hv.Histogram((frequencies, edges))[..., 'Non-existent']