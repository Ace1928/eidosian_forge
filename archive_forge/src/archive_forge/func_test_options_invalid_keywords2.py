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
def test_options_invalid_keywords2(self):
    try:
        Options('test', allowed_keywords=['kw2'], kw2='value', kw3='value')
    except OptionError as e:
        self.assertEqual(str(e), "Invalid option 'kw3', valid options are: ['kw2'].")