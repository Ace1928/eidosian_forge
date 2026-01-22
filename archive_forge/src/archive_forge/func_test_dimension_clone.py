import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_dimension_clone(self):
    dim1 = Dimension('Test dimension')
    dim2 = dim1.clone(cyclic=True)
    self.assertEqual(dim2.cyclic, True)
    dim3 = dim1.clone('New test dimension', unit='scovilles')
    self.assertEqual(dim3.name, 'New test dimension')
    self.assertEqual(dim3.unit, 'scovilles')