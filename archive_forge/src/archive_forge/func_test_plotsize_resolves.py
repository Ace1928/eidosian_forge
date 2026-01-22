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
def test_plotsize_resolves(self):
    points = Points([1, 2, 3])
    PlotSize(source=points)
    plot = bokeh_server_renderer.get_plot(points)
    callback = plot.callbacks[0]
    model = namedtuple('Plot', 'inner_width inner_height ref')(400, 300, {'id': 'Test'})
    width_spec = callback.attributes['width']
    height_spec = callback.attributes['height']
    resolved = callback.resolve_attr_spec(width_spec, model, model=model)
    self.assertEqual(resolved, {'id': 'Test', 'value': 400})
    resolved = callback.resolve_attr_spec(height_spec, model, model=model)
    self.assertEqual(resolved, {'id': 'Test', 'value': 300})