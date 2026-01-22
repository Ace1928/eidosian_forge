import numpy as np
from bokeh.layouts import Column
from bokeh.models import Div, Toolbar
from holoviews.core import (
from holoviews.element import Curve, Image, Points
from holoviews.operation import gridmatrix
from holoviews.streams import Stream
from .test_plot import TestBokehPlot, bokeh_renderer
def test_gridmatrix_overlaid_batched(self):
    ds = Dataset((['A'] * 5 + ['B'] * 5, np.random.rand(10), np.random.rand(10)), kdims=['a', 'b', 'c'])
    gmatrix = gridmatrix(ds.groupby('a', container_type=NdOverlay))
    plot = bokeh_renderer.get_plot(gmatrix)
    sp1 = plot.subplots['b', 'c']
    self.assertEqual(sp1.state.xaxis[0].visible, False)
    self.assertEqual(sp1.state.yaxis[0].visible, True)
    sp2 = plot.subplots['b', 'b']
    self.assertEqual(sp2.state.xaxis[0].visible, True)
    self.assertEqual(sp2.state.yaxis[0].visible, True)
    sp3 = plot.subplots['c', 'b']
    self.assertEqual(sp3.state.xaxis[0].visible, True)
    self.assertEqual(sp3.state.yaxis[0].visible, False)
    sp4 = plot.subplots['c', 'c']
    self.assertEqual(sp4.state.xaxis[0].visible, False)
    self.assertEqual(sp4.state.yaxis[0].visible, False)