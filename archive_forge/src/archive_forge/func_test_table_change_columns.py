from datetime import datetime as dt
from bokeh.models.widgets import (
from holoviews.core.options import Store
from holoviews.core.spaces import DynamicMap
from holoviews.element import Table
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.callbacks import CDSCallback
from holoviews.plotting.bokeh.renderer import BokehRenderer
from holoviews.streams import CDSStream, Stream
def test_table_change_columns(self):
    lengths = {'a': 1, 'b': 2, 'c': 3}
    table = DynamicMap(lambda a: Table(range(lengths[a]), a), kdims=['a']).redim.values(a=['a', 'b', 'c'])
    plot = bokeh_renderer.get_plot(table)
    self.assertEqual(sorted(plot.handles['source'].data.keys()), ['a'])
    self.assertEqual(plot.handles['table'].columns[0].title, 'a')
    plot.update(('b',))
    self.assertEqual(sorted(plot.handles['source'].data.keys()), ['b'])
    self.assertEqual(plot.handles['table'].columns[0].title, 'b')