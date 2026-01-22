import pytest
import holoviews as hv
from holoviews.core import Store
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.bokeh.styles import expand_batched_style
from holoviews.plotting.bokeh.util import (
def test_glyph_order(self):
    order = glyph_order(['scatter_1', 'patch_1', 'rect_1'], ['scatter', 'patch'])
    self.assertEqual(order, ['scatter_1', 'patch_1', 'rect_1'])