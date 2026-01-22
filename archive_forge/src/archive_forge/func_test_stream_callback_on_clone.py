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
def test_stream_callback_on_clone(self):
    points = Points([])
    stream = PointerXY(source=points)
    plot = bokeh_server_renderer.get_plot(points.clone())
    bokeh_server_renderer(plot)
    set_curdoc(plot.document)
    plot.callbacks[0].on_msg({'x': 0.8, 'y': 0.3})
    self.assertEqual(stream.x, 0.8)
    self.assertEqual(stream.y, 0.3)