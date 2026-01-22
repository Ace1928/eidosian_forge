import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
def test_expand_style_opts_multiple(self):
    style = {'line_color': 'red', 'line_width': 4}
    opts = ['line_color', 'line_width']
    data, mapping = expand_batched_style(style, opts, {}, nvals=3)
    self.assertEqual(data['line_color'], ['red', 'red', 'red'])
    self.assertEqual(data['line_width'], [4, 4, 4])
    self.assertEqual(mapping, {'line_color': {'field': 'line_color'}, 'line_width': {'field': 'line_width'}})