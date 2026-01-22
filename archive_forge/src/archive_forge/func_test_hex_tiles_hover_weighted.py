import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_hover_weighted(self):
    tiles = HexTiles([(0, 0, 0.1), (0.5, 0.5, 0.2), (-0.5, -0.5, 0.3)], vdims='z').opts(aggregator=np.mean)
    plot = bokeh_renderer.get_plot(tiles)
    dims, opts = plot._hover_opts(tiles)
    self.assertEqual(dims, [Dimension('z')])
    self.assertEqual(opts, {})