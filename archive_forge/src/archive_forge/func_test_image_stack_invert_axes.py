from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_image_stack_invert_axes(self):
    x = np.arange(self.xsize)
    y = np.arange(self.ysize) + 5
    a, b, c = (self.a, self.b, self.c)
    img_stack = ImageStack((x, y, a, b, c), kdims=['x', 'y'], vdims=['a', 'b', 'c'])
    plot = bokeh_renderer.get_plot(img_stack.opts(invert_axes=True))
    source = plot.handles['source']
    np.testing.assert_equal(source.data['image'][0][:, :, 0].T, a)
    np.testing.assert_equal(source.data['image'][0][:, :, 1].T, b)
    np.testing.assert_equal(source.data['image'][0][:, :, 2].T, c)
    assert source.data['x'][0] == 4.5
    assert source.data['y'][0] == -0.5
    assert source.data['dw'][0] == self.ysize
    assert source.data['dh'][0] == self.xsize