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
def test_point_draw_shared_datasource_callback(self):
    points = Points([1, 2, 3])
    table = Table(points.data, ['x', 'y'])
    layout = (points + table).opts(shared_datasource=True, clone=False)
    PointDraw(source=points)
    self.assertIs(points.data, table.data)
    plot = bokeh_renderer.get_plot(layout)
    point_plot = plot.subplots[0, 0].subplots['main']
    table_plot = plot.subplots[0, 1].subplots['main']
    self.assertIs(point_plot.handles['source'], table_plot.handles['source'])