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
def clear_options(self):
    Store.options(val=OptionTree(groups=['plot', 'style'], backend='matplotlib'), backend='matplotlib')
    Store.options(val=OptionTree(groups=['plot', 'style'], backend='bokeh'), backend='bokeh')
    Store.custom_options({}, backend='matplotlib')
    Store.custom_options({}, backend='bokeh')