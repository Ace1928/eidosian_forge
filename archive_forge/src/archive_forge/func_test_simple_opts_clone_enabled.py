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
def test_simple_opts_clone_enabled(self):
    im = Image(np.random.rand(10, 10))
    styled_im = im.opts(interpolation='nearest', cmap='jet', clone=True)
    self.assertEqual(self.lookup_options(im, 'plot').options, {})
    self.assertEqual(self.lookup_options(styled_im, 'plot').options, {})
    assert styled_im is not im
    im_lookup = self.lookup_options(im, 'style').options
    self.assertEqual(im_lookup['cmap'] == 'jet', False)
    styled_im_lookup = self.lookup_options(styled_im, 'style').options
    self.assertEqual(styled_im_lookup['cmap'] == 'jet', True)