from datetime import datetime as dt
from bokeh.models.widgets import (
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream
def test_table_selected(self):
    table = Table([(0, 0), (1, 1), (2, 2)], ['x', 'y']).opts(selected=[0, 2])
    plot = bokeh_renderer.get_plot(table)
    cds = plot.handles['cds']
    self.assertEqual(cds.selected.indices, [0, 2])