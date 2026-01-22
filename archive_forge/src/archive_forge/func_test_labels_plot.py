import unittest
import numpy as np
import holoviews as hv
from holoviews.element import (
from holoviews.plotting.bokeh.util import bokeh32, bokeh33
from .test_plot import TestBokehPlot, bokeh_renderer
def test_labels_plot(self):
    text = Labels([(0, 0, 'Test')])
    plot = bokeh_renderer.get_plot(text)
    source = plot.handles['source']
    data = {'x': np.array([0]), 'y': np.array([0]), 'Label': ['Test']}
    for c, col in source.data.items():
        self.assertEqual(col, data[c])