import numpy as np
import holoviews as hv
from holoviews.element.comparison import ComparisonTestCase
def test_table_ellipsis_slice_value_weight(self):
    sliced = self.table[..., 'Weight']
    assert sliced.vdims == ['Weight']