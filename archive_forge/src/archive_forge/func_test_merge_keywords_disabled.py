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
def test_merge_keywords_disabled(self):
    options = self.initialize_option_tree()
    options.Image = Options('style', clims=(0, 0.5), merge_keywords=False)
    expected = {'clims': (0, 0.5)}
    direct_kws = options.Image.groups['style'].kwargs
    inherited_kws = options.Image.options('style').kwargs
    self.assertEqual(direct_kws, expected)
    self.assertEqual(inherited_kws, expected)