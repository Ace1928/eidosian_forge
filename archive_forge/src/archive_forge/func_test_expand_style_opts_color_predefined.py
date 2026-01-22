import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
def test_expand_style_opts_color_predefined(self):
    style = {'fill_color': 'red'}
    opts = ['color', 'line_color', 'fill_color']
    data, mapping = expand_batched_style(style, opts, {'color': 'color'}, nvals=3)
    self.assertEqual(data['fill_color'], ['red', 'red', 'red'])
    self.assertEqual(mapping, {'fill_color': {'field': 'fill_color'}})