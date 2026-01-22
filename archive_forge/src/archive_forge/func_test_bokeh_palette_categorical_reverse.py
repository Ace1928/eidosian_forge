import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_bokeh_palette_categorical_reverse(self):
    colors = bokeh_palette_to_palette('Category20_r', 3)
    self.assertEqual(colors, ['#1f77b4', '#8c564b', '#9edae5'][::-1])