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
def test_points_histogram_overwrite_intersect_hide_region(self, dynamic=False):
    self.do_crossfilter_points_histogram(selection_mode='overwrite', cross_filter_mode='intersect', selected1=[1, 2], selected2=[1], selected3=[0], selected4=[2], points_region1=[], points_region2=[], points_region3=[], points_region4=[], hist_region2=[], hist_region3=[], hist_region4=[], show_regions=False, dynamic=dynamic)