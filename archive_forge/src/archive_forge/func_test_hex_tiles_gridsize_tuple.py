import numpy as np
from holoviews.core import Dimension
from holoviews.element import HexTiles
from holoviews.plotting.bokeh.hex_tiles import hex_binning
from holoviews.plotting.bokeh.util import property_to_dict
from .test_plot import TestBokehPlot, bokeh_renderer
def test_hex_tiles_gridsize_tuple(self):
    tiles = HexTiles([(0, 0), (0.5, 0.5), (-0.5, -0.5), (-0.4, -0.4)]).opts(gridsize=(5, 10))
    plot = bokeh_renderer.get_plot(tiles)
    glyph = plot.handles['glyph']
    self.assertEqual(glyph.size, 0.06666666666666667)
    self.assertEqual(glyph.aspect_scale, 0.5)