from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_nodata_array_uint(self):
    img = Image(np.array([[0, 1], [2, 0]], dtype='uint32')).opts(nodata=0)
    plot = bokeh_renderer.get_plot(img)
    cmapper = plot.handles['color_mapper']
    source = plot.handles['source']
    assert cmapper.low == 1
    assert cmapper.high == 2
    np.testing.assert_equal(source.data['image'][0], np.array([[2, np.nan], [np.nan, 1]]))