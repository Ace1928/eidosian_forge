from collections import defaultdict
from unittest import SkipTest
import pandas as pd
import param
import pytest
from panel.widgets import IntSlider
import holoviews as hv
from holoviews.core.spaces import DynamicMap
from holoviews.core.util import Version
from holoviews.element import Curve, Histogram, Points, Polygons, Scatter
from holoviews.element.comparison import ComparisonTestCase
from holoviews.streams import *  # noqa (Test all available streams)
from holoviews.util import Dynamic, extension
from holoviews.util.transform import dim
from .utils import LoggingComparisonTestCase
@pytest.mark.usefixtures('bokeh_backend')
def test_dynamicmap_partial_bind_and_streams():

    def make_plot(z, x_range, y_range):
        return Curve([1, 2, 3, 4, z])
    slider = IntSlider(name='Slider', start=0, end=10)
    range_xy = RangeXY()
    dmap = DynamicMap(param.bind(make_plot, z=slider), streams=[range_xy])
    bk_figure = hv.render(dmap)
    assert bk_figure.renderers[0].data_source.data['y'][-1] == 0
    assert range_xy.x_range == (0, 4)
    assert range_xy.y_range == (-0.4, 4.4)