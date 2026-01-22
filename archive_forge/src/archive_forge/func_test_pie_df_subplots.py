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
def test_pie_df_subplots(self):
    df = DataFrame(np.random.default_rng(2).random((5, 3)), columns=['X', 'Y', 'Z'], index=['a', 'b', 'c', 'd', 'e'])
    axes = _check_plot_works(df.plot.pie, default_axes=True, subplots=True)
    assert len(axes) == len(df.columns)
    for ax in axes:
        _check_text_labels(ax.texts, df.index)
    for ax, ylabel in zip(axes, df.columns):
        assert ax.get_ylabel() == ylabel