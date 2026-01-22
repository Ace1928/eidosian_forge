from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_stack_array(self):
    a, b, c = (self.a, self.b, self.c)
    data = np.dstack((a, b, c))
    img_stack = ImageStack(data, kdims=['x', 'y'], vdims=['a', 'b', 'c'])
    plot = bokeh_renderer.get_plot(img_stack)
    source = plot.handles['source']
    np.testing.assert_equal(source.data['image'][0][:, :, 0], a)
    np.testing.assert_equal(source.data['image'][0][:, :, 1], b)
    np.testing.assert_equal(source.data['image'][0][:, :, 2], c)
    assert source.data['x'][0] == -0.5
    assert source.data['y'][0] == -0.5
    assert source.data['dw'][0] == self.xsize
    assert source.data['dh'][0] == self.ysize
    assert isinstance(plot, ImageStackPlot)