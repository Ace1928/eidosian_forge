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
def test_selection1d_resolves_table(self):
    table = Table([1, 2, 3], 'x')
    Selection1D(source=table)
    plot = bokeh_server_renderer.get_plot(table)
    selected = Selection(indices=[0, 2])
    callback = plot.callbacks[0]
    spec = callback.attributes['index']
    resolved = callback.resolve_attr_spec(spec, selected, model=selected)
    self.assertEqual(resolved, {'id': selected.ref['id'], 'value': [0, 2]})