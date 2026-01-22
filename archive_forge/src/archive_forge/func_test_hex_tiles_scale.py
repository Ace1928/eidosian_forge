import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_scale(self):
    tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).opts(size_index=2, gridsize=3)
    plot = bokeh_renderer.get_plot(tiles)
    source = plot.handles['source']
    self.assertEqual(source.data['scale'], np.array([0.45, 0.45, 0.9]))