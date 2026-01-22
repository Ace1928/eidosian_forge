from itertools import product
import numpy as np
from bokeh.models import ColorBar
from holoviews.core.spaces import HoloMap
from holoviews.element.raster import HeatMap
from holoviews.plotting.bokeh import RadialHeatMapPlot
from .test_plot import TestBokehPlot, bokeh_renderer
def test_compute_seg_tick_mappings(self):
    """Test computation of segment tick mappings. Check integers, list and
        function types.

        """
    order = sorted(self.seg_bins.keys())
    self.plot.xticks = 1
    ticks = self.plot._compute_tick_mapping('angle', order, self.seg_bins)
    self.assertEqual(ticks, {'o1': self.seg_bins['o1']})
    self.plot.xticks = 2
    ticks = self.plot._compute_tick_mapping('angle', order, self.seg_bins)
    self.assertEqual(ticks, self.seg_bins)
    self.plot.xticks = ['New Tick1', 'New Tick2']
    ticks = self.plot._compute_tick_mapping('angle', order, self.seg_bins)
    bins = self.plot._get_bins('angle', self.plot.xticks, True)
    ticks_cmp = {x: bins[x] for x in self.plot.xticks}
    self.assertEqual(ticks, ticks_cmp)
    self.plot.xticks = lambda x: x == 'o1'
    ticks = self.plot._compute_tick_mapping('angle', order, self.seg_bins)
    self.assertEqual(ticks, {'o1': self.seg_bins['o1']})