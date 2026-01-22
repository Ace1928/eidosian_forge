import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_columns_collapse_heterogeneous(self):
    collapsed = HoloMap({i: Dataset({'x': self.xs, 'y': self.ys * i}, kdims=['x'], vdims=['y']) for i in range(10)}, kdims=['z']).collapse('z', np.mean)
    expected = Dataset({'x': self.xs, 'y': self.ys * 4.5}, kdims=['x'], vdims=['y'])
    self.compare_dataset(collapsed, expected)