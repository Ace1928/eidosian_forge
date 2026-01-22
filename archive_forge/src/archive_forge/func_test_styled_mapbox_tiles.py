import numpy as np
import pytest
from holoviews.element import RGB, Bounds, Points, Tiles
from holoviews.element.tiles import _ATTRIBUTIONS, StamenTerrain
from .test_plot import TestPlotlyPlot, plotly_renderer
def test_styled_mapbox_tiles(self):
    tiles = Tiles().opts(mapboxstyle='dark', accesstoken='token-str').redim.range(x=self.x_range, y=self.y_range)
    fig_dict = plotly_renderer.get_plot_state(tiles)
    subplot = fig_dict['layout']['mapbox']
    self.assertEqual(subplot['style'], 'dark')
    self.assertEqual(subplot['accesstoken'], 'token-str')
    self.assertEqual(subplot['center'], {'lat': self.lat_center, 'lon': self.lon_center})