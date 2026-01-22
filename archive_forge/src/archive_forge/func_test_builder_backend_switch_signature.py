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
def test_builder_backend_switch_signature(self):
    Store.options(val=self.store_mpl, backend='matplotlib')
    Store.options(val=self.store_bokeh, backend='bokeh')
    Store.set_current_backend('bokeh')
    self.assertEqual(opts.Curve.__signature__ is not None, True)
    sigkeys = opts.Curve.__signature__.parameters
    self.assertEqual('color' in sigkeys, True)
    self.assertEqual('line_width' in sigkeys, True)
    Store.set_current_backend('matplotlib')
    self.assertEqual(opts.Curve.__signature__ is not None, True)
    sigkeys = opts.Curve.__signature__.parameters
    self.assertEqual('color' in sigkeys, True)
    self.assertEqual('linewidth' in sigkeys, True)