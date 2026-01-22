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
def test_mpl_bokeh_mpl(self):
    img = Image(np.random.rand(10, 10))
    Store.current_backend = 'matplotlib'
    StoreOptions.set_options(img, style={'Image': {'cmap': 'Blues'}})
    mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
    self.assertEqual(mpl_opts, {'cmap': 'Blues'})
    Store.current_backend = 'bokeh'
    StoreOptions.set_options(img, style={'Image': {'cmap': 'Purple'}})
    bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
    self.assertEqual(bokeh_opts, {'cmap': 'Purple'})
    Store.current_backend = 'matplotlib'
    mpl_opts = Store.lookup_options('matplotlib', img, 'style').options
    self.assertEqual(mpl_opts, {'cmap': 'Blues'})
    Store.current_backend = 'bokeh'
    bokeh_opts = Store.lookup_options('bokeh', img, 'style').options
    self.assertEqual(bokeh_opts, {'cmap': 'Purple'})