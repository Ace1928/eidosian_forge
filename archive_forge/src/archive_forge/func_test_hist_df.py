from datetime import (
import gc
import itertools
import re
import string
import weakref
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.api import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_hist_df(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((100, 4)))
    ax = _check_plot_works(df.plot.hist)
    expected = [pprint_thing(c) for c in df.columns]
    _check_legend_labels(ax, labels=expected)
    axes = _check_plot_works(df.plot.hist, default_axes=True, subplots=True, logy=True)
    _check_axes_shape(axes, axes_num=4, layout=(4, 1))
    _check_ax_scales(axes, yaxis='log')