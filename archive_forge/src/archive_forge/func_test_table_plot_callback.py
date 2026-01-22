from datetime import datetime as dt
from bokeh.models.widgets import (
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream
def test_table_plot_callback(self):
    table = Table(([1, 2, 3], [1.0, 2.0, 3.0], ['A', 'B', 'C']), ['x', 'y'], 'z')
    CDSStream(source=table)
    plot = bokeh_renderer.get_plot(table)
    self.assertEqual(len(plot.callbacks), 1)
    self.assertIsInstance(plot.callbacks[0], CDSCallback)