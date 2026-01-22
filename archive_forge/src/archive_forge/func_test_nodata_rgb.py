from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
from bokeh.models import CustomJSHover, HoverTool
from holoviews.element import RGB, Image, ImageStack, Raster
from holoviews.plotting.bokeh.raster import ImageStackPlot
from holoviews.plotting.bokeh.util import bokeh34
from .test_plot import TestBokehPlot, bokeh_renderer
def test_nodata_rgb(self):
    N = 2
    rgb_d = np.linspace(0, 1, N * N * 3).reshape(N, N, 3)
    rgb = RGB(rgb_d).redim.nodata(R=0)
    plot = bokeh_renderer.get_plot(rgb)
    image_data = plot.handles['source'].data['image'][0]
    assert (image_data == 0).sum() == 1