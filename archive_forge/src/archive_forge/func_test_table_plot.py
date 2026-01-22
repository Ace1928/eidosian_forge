from datetime import datetime as dt
from bokeh.models.widgets import (
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream
def test_table_plot(self):
    table = Table(([1, 2, 3], [1.0, 2.0, 3.0], ['A', 'B', 'C']), ['x', 'y'], 'z')
    plot = bokeh_renderer.get_plot(table)
    dims = table.dimensions()
    formatters = (NumberFormatter, NumberFormatter, StringFormatter)
    editors = (IntEditor, NumberEditor, StringEditor)
    for dim, fmt, edit, column in zip(dims, formatters, editors, plot.state.columns):
        self.assertEqual(column.title, dim.pprint_label)
        self.assertIsInstance(column.formatter, fmt)
        self.assertIsInstance(column.editor, edit)