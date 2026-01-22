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
def initialize_option_tree(self):
    Store.options(val=OptionTree(groups=['plot', 'style']))
    options = Store.options()
    options.Image = Options('style', cmap='hot', interpolation='nearest')
    return options