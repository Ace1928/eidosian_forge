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
def test_optiontree_dict_setter_getter(self):
    options = OptionTree(groups=['group1', 'group2'])
    opts1 = Options(kw1='value1')
    opts2 = Options(kw2='value2')
    options.MyType = {'group1': opts1, 'group2': opts2}
    self.assertEqual(options.MyType['group1'], opts1)
    self.assertEqual(options.MyType['group1'].options, {'kw1': 'value1'})
    self.assertEqual(options.MyType['group2'], opts2)
    self.assertEqual(options.MyType['group2'].options, {'kw2': 'value2'})