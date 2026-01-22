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
def test_cds_resolves(self):
    points = Points([1, 2, 3])
    CDSStream(source=points)
    plot = bokeh_server_renderer.get_plot(points)
    cds = plot.handles['cds']
    callback = plot.callbacks[0]
    data_spec = callback.attributes['data']
    resolved = callback.resolve_attr_spec(data_spec, cds, model=cds)
    self.assertEqual(resolved, {'id': cds.ref['id'], 'value': points.columns()})