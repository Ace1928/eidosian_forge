import os
import pickle
import numpy as np
import pytest
from holoviews import (
from holoviews.core.options import (
from holoviews.element.comparison import ComparisonTestCase
from holoviews.plotting import mpl # noqa
from holoviews.plotting import bokeh # noqa
from holoviews.plotting import plotly # noqa
def test_mpl_bokeh_mpl_via_dict_backend_keyword(self):
    curve = Curve([1, 2, 3])
    styled_mpl = curve.opts({'Curve': dict(color='red')}, backend='matplotlib')
    styled = styled_mpl.opts({'Curve': dict(color='green')}, backend='bokeh')
    mpl_lookup = Store.lookup_options('matplotlib', styled, 'style')
    self.assertEqual(mpl_lookup.kwargs['color'], 'red')
    bokeh_lookup = Store.lookup_options('bokeh', styled, 'style')
    self.assertEqual(bokeh_lookup.kwargs['color'], 'green')