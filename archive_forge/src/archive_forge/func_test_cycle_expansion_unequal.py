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
def test_cycle_expansion_unequal(self):
    cycle1 = Cycle(values=['a', 'b', 'c', 'd'])
    cycle2 = Cycle(values=[1, 2, 3])
    opts = Options('test', one=cycle1, two=cycle2)
    self.assertEqual(opts[0], {'one': 'a', 'two': 1})
    self.assertEqual(opts[1], {'one': 'b', 'two': 2})
    self.assertEqual(opts[2], {'one': 'c', 'two': 3})
    self.assertEqual(opts[3], {'one': 'd', 'two': 1})
    self.assertEqual(opts[4], {'one': 'a', 'two': 2})
    self.assertEqual(opts[5], {'one': 'b', 'two': 3})