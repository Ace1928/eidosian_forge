import datetime as dt
import re
import numpy as np
from bokeh.models import Div, GlyphRenderer, GridPlot, Spacer, Tabs, Title, Toolbar
from bokeh.models.layouts import TabPanel
from bokeh.plotting import figure
from holoviews.core import (
from holoviews.element import Curve, Histogram, Image, Points, Scatter
from holoviews.streams import Stream
from holoviews.util import opts, render
from holoviews.util.transform import dim
from ...utils import LoggingComparisonTestCase
from .test_plot import TestBokehPlot, bokeh_renderer
def test_layout_shared_source_synced_update(self):
    hmap = HoloMap({i: Dataset({chr(65 + j): np.random.rand(i + 2) for j in range(4)}, kdims=['A', 'B', 'C', 'D']) for i in range(3)})
    hmap1 = hmap.map(lambda x: Points(x.clone(kdims=['A', 'B'])), Dataset)
    hmap2 = hmap.map(lambda x: Points(x.clone(kdims=['D', 'C'])), Dataset)
    hmap2.pop((1,))
    layout = (hmap1 + hmap2).opts(shared_datasource=True)
    plot = bokeh_renderer.get_plot(layout)
    sources = plot.handles.get('shared_sources', [])
    source_cols = plot.handles.get('source_cols', {})
    self.assertEqual(len(sources), 1)
    source = sources[0]
    data = source.data
    cols = source_cols[id(source)]
    self.assertEqual(set(cols), {'A', 'B', 'C', 'D'})
    self.assertEqual(set(data.keys()), {'A', 'B', 'C', 'D'})
    plot.update((1,))
    self.assertEqual(data['A'], hmap1[1].dimension_values(0))
    self.assertEqual(data['B'], hmap1[1].dimension_values(1))
    self.assertEqual(data['C'], np.full_like(hmap1[1].dimension_values(0), np.nan))
    self.assertEqual(data['D'], np.full_like(hmap1[1].dimension_values(0), np.nan))