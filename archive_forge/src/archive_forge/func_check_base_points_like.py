from unittest import skip, skipIf
import pandas as pd
import panel as pn
import holoviews as hv
from holoviews.core.options import Cycle, Store
from holoviews.element import ErrorBars, Points, Rectangles, Table, VSpan
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting.util import linear_gradient
from holoviews.selection import link_selections
from holoviews.streams import SelectionXY
def check_base_points_like(self, base_points, lnk_sel, data=None):
    if data is None:
        data = self.data
    self.assertEqual(self.element_color(base_points), lnk_sel.unselected_color)
    self.assertEqual(base_points.data, data)