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
def test_backend_opts_to_default_inheritance(self):
    """
        Checks customs inheritance backs off to default tree correctly
        using .opts.
        """
    options = self.initialize_option_tree()
    options.Image.A.B = Options('style', alpha=0.2)
    obj = Image(np.random.rand(10, 10), group='A', label='B')
    expected_obj = {'alpha': 0.2, 'cmap': 'hot', 'interpolation': 'nearest'}
    obj_lookup = Store.lookup_options('matplotlib', obj, 'style')
    self.assertEqual(obj_lookup.kwargs, expected_obj)
    custom_obj = opts.apply_groups(obj, style=dict(clims=(0, 0.5)))
    expected_custom_obj = dict(clims=(0, 0.5), **expected_obj)
    custom_obj_lookup = Store.lookup_options('matplotlib', custom_obj, 'style')
    self.assertEqual(custom_obj_lookup.kwargs, expected_custom_obj)