import numpy as np
from holoviews.core.spaces import DynamicMap
from holoviews.element import Spread
from holoviews.streams import Buffer
from .test_plot import TestBokehPlot, bokeh_renderer
def test_spread_with_nans(self):
    spread = Spread([(0, 0, 0, 1), (1, 0, 0, 2), (2, 0, 0, 3), (3, np.nan, np.nan, np.nan), (4, 0, 0, 5), (5, 0, 0, 6), (6, 0, 0, 7)], vdims=['y', 'neg', 'pos'])
    plot = bokeh_renderer.get_plot(spread)
    cds = plot.handles['cds']
    self.assertEqual(cds.data['x'], np.array([0.0, 1.0, 2.0, 2.0, 1.0, 0.0, np.nan, 4.0, 5.0, 6.0, 6.0, 5.0, 4.0]))
    self.assertEqual(cds.data['y'], np.array([0.0, 0.0, 0.0, 3.0, 2.0, 1.0, np.nan, 0.0, 0.0, 0.0, 7.0, 6.0, 5.0]))