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
def test_layout_title_format(self):
    title_str = 'Label: {label}, group: {group}, dims: {dimensions}, type: {type}'
    layout = NdLayout({'Element 1': Scatter([], label='ONE', group='first'), 'Element 2': Scatter([], label='TWO', group='second')}, kdims='MYDIM', label='the_label', group='the_group').opts(opts.NdLayout(title=title_str), opts.Scatter(title=title_str))
    title = bokeh_renderer.get_plot(layout).handles['title']
    self.assertIsInstance(title, Div)
    text = 'Label: the_label, group: the_group, dims: , type: NdLayout'
    self.assertEqual(re.split('>|</', title.text)[1], text)
    plot = render(layout)
    titles = {title.text for title in list(plot.select({'type': Title}))}
    titles_correct = {'Label: ONE, group: first, dims: MYDIM: Element 1, type: Scatter', 'Label: TWO, group: second, dims: MYDIM: Element 2, type: Scatter'}
    self.assertEqual(titles_correct, titles)