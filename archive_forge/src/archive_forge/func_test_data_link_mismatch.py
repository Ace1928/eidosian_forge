import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_data_link_mismatch(self):
    polys = Polygons([np.random.rand(10, 2)])
    table = Table([('A', 1), ('B', 2)], 'A', 'B')
    DataLink(polys, table)
    layout = polys + table
    msg = 'DataLink source data length must match target'
    with pytest.raises(ValueError, match=msg):
        bokeh_renderer.get_plot(layout)