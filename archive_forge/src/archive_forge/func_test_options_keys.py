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
def test_options_keys(self):
    opts = Options('test', allowed_keywords=['kw3', 'kw2'], kw2='value', kw3='value')
    self.assertEqual(opts.keys(), ['kw2', 'kw3'])