from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_get_markers(self):
    """Test computation of marker positions for function, list, tuple and
        integer type.

        """
    args = [sorted(self.ann_bins.keys()), self.ann_bins]
    test_val = np.array(self.ann_bins['o1'][1])
    test_input = lambda x: x == 'o1'
    self.assertEqual(self.plot._get_markers(test_input, *args), test_val)
    test_val = np.array(self.ann_bins['o2'][1])
    test_input = [1]
    self.assertEqual(self.plot._get_markers(test_input, *args), test_val)
    test_val = np.array(self.ann_bins['o2'][1])
    test_input = ('o2',)
    self.assertEqual(self.plot._get_markers(test_input, *args), test_val)
    test_val = np.array(self.ann_bins['o1'][1])
    test_input = 1
    self.assertEqual(self.plot._get_markers(test_input, *args), test_val)