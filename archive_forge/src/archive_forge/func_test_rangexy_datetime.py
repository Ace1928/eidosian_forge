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
def test_rangexy_datetime(self):
    df = pd.DataFrame(data=np.random.default_rng(2).standard_normal((30, 4)), columns=list('ABCD'), index=pd.date_range('2018-01-01', freq='D', periods=30))
    curve = Curve(df, 'index', 'C')
    stream = RangeXY(source=curve)
    plot = bokeh_server_renderer.get_plot(curve)
    callback = plot.callbacks[0]
    callback.on_msg({'x0': curve.iloc[0, 0], 'x1': curve.iloc[3, 0], 'y0': 0.2, 'y1': 0.8})
    self.assertEqual(stream.x_range[0], curve.iloc[0, 0])
    self.assertEqual(stream.x_range[1], curve.iloc[3, 0])
    self.assertEqual(stream.y_range, (0.2, 0.8))