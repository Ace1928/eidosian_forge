import numpy as np
import pytest
from bokeh.models import ColumnDataSource, RangeTool
from holoviews.core.spaces import DynamicMap
from holoviews.element import Curve, Image, Path, Points, Polygons, Scatter, Table
from holoviews.plotting.links import DataLink, Link, RangeToolLink
from .test_plot import TestBokehPlot, bokeh_renderer
def test_data_link_poly_table_on_unlinked_clone(self):
    arr1 = np.random.rand(10, 2)
    arr2 = np.random.rand(10, 2)
    polys = Polygons([arr1, arr2])
    table = Table([('A', 1), ('B', 2)], 'A', 'B')
    DataLink(polys, table)
    layout = polys.clone() + table.clone(link=False)
    plot = bokeh_renderer.get_plot(layout)
    cds = list(plot.state.select({'type': ColumnDataSource}))
    self.assertEqual(len(cds), 2)