import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tile_alpha_op(self):
    hextiles = HexTiles(np.random.randn(1000, 2)).opts(alpha='Count')
    plot = bokeh_renderer.get_plot(hextiles)
    glyph = plot.handles['glyph']
    self.assertEqual(property_to_dict(glyph.fill_alpha), {'field': 'alpha'})