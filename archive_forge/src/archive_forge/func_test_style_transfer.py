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
def test_style_transfer(self):
    hist = opts.apply_groups(self.hist, style={'style1': 'style_child'})
    hist2 = self.hist.opts()
    opts_kwargs = Store.lookup_options('matplotlib', hist2, 'style').kwargs
    self.assertEqual(opts_kwargs, {'style1': 'style1', 'style2': 'style2'})
    Store.transfer_options(hist, hist2, 'matplotlib')
    opts_kwargs = Store.lookup_options('matplotlib', hist2, 'style').kwargs
    self.assertEqual(opts_kwargs, {'style1': 'style_child', 'style2': 'style2'})