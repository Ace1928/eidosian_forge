import pandas as pd
from bokeh.models import FactorRange
from holoviews.core import NdOverlay
from holoviews.element import Segments
from .test_plot import TestBokehPlot, bokeh_renderer
def test_segments_overlay_datetime_hover(self):
    obj = NdOverlay({i: Segments((list(pd.date_range('2016-01-01', '2016-01-31')), range(31), pd.date_range('2016-01-02', '2016-02-01'), range(31))) for i in range(5)}, kdims=['Test']).opts({'Segments': {'tools': ['hover']}})
    tooltips = [('Test', '@{Test}'), ('x0', '@{x0}{%F %T}'), ('y0', '@{y0}'), ('x1', '@{x1}{%F %T}'), ('y1', '@{y1}')]
    formatters = {'@{x0}': 'datetime', '@{x1}': 'datetime'}
    self._test_hover_info(obj, tooltips, formatters=formatters)