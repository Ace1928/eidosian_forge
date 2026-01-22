import datetime as dt
from collections import deque, namedtuple
from unittest import SkipTest
import numpy as np
import pandas as pd
import pytest
import pyviz_comms as comms
from bokeh.events import Tap
from bokeh.io.doc import set_curdoc
from bokeh.models import ColumnDataSource, Plot, PolyEditTool, Range1d, Selection
from holoviews.core import DynamicMap
from holoviews.core.options import Store
from holoviews.element import Box, Curve, Points, Polygons, Rectangles, Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import (
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import (
def test_rangexy_multi_yaxes():
    c1 = Curve(np.arange(100).cumsum(), vdims='y')
    c2 = Curve(-np.arange(100).cumsum(), vdims='y2')
    RangeXY(source=c1)
    RangeXY(source=c2)
    overlay = (c1 * c2).opts(backend='bokeh', multi_y=True)
    plot = bokeh_server_renderer.get_plot(overlay)
    p1, p2 = plot.subplots.values()
    assert plot.state.y_range is p1.handles['y_range']
    assert 'y2' in plot.state.extra_y_ranges
    assert plot.state.extra_y_ranges['y2'] is p2.handles['y_range']
    assert p1.callbacks[0].plot is p1
    assert p2.callbacks[0].plot is p2