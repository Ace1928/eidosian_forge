import numpy as np
from holoviews import Dimension, NdOverlay, Overlay
from holoviews.core.options import Cycle, Store
from holoviews.core.spaces import DynamicMap, HoloMap
from holoviews.element import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.operation import operation
from holoviews.plotting.util import (
from holoviews.streams import PointerX
def test_bokeh_palette_categorical_palettes_not_interpolated(self):
    categorical = ('accent', 'category20', 'dark2', 'colorblind', 'pastel1', 'pastel2', 'set1', 'set2', 'set3', 'paired')
    for cat in categorical:
        self.assertTrue(len(set(bokeh_palette_to_palette(cat))) <= 20)