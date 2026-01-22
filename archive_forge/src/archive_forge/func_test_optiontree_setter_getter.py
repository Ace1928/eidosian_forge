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
def test_optiontree_setter_getter(self):
    options = OptionTree(groups=['group1', 'group2'])
    opts = Options('group1', kw1='value')
    options.MyType = opts
    self.assertEqual(options.MyType['group1'], opts)
    self.assertEqual(options.MyType['group1'].options, {'kw1': 'value'})