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
def test_opts_method_with_utility(self):
    im = Image(np.random.rand(10, 10))
    imopts = opts.Image(cmap='Blues')
    styled_im = im.opts(imopts)
    assert styled_im is im
    self.assertEqual(self.lookup_options(im, 'style').options, {'cmap': 'Blues', 'interpolation': 'nearest'})