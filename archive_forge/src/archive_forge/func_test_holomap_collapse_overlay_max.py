import numpy as np
from holoviews import Dataset, HoloMap
from holoviews.core import Dimension
from holoviews.core.ndmapping import (
from holoviews.core.overlay import Overlay
from holoviews.element import Curve
from holoviews.element.comparison import ComparisonTestCase
def test_holomap_collapse_overlay_max(self):
    hmap = HoloMap({(1, 0): Curve(np.arange(8)) * Curve(-np.arange(8)), (2, 0): Curve(np.arange(8) ** 2) * Curve(-np.arange(8) ** 3)}, kdims=['A', 'B'])
    self.assertEqual(hmap.collapse(function=np.max), Overlay([(('Curve', 'I'), Curve({'x': np.arange(8), 'y': np.arange(8) ** 2}, kdims=['x'], vdims=['y'])), (('Curve', 'II'), Curve({'x': np.arange(8), 'y': -np.arange(8)}, kdims=['x'], vdims=['y']))]))