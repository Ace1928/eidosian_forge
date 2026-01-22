import numpy as np
from bokeh.models import FactorRange, HoverTool, Range1d
from holoviews.element import HeatMap, Image, Points
from .test_plot import TestBokehPlot, bokeh_renderer
def test_heatmap_colormapping(self):
    hm = HeatMap([(1, 1, 1), (2, 2, 0)])
    self._test_colormapping(hm, 2)