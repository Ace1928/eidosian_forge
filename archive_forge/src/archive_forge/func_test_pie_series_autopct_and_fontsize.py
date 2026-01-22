from datetime import datetime
from itertools import chain
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
def test_pie_series_autopct_and_fontsize(self):
    series = Series(np.random.default_rng(2).integers(1, 5), index=['a', 'b', 'c', 'd', 'e'], name='YLABEL')
    color_args = ['r', 'g', 'b', 'c', 'm']
    ax = _check_plot_works(series.plot.pie, colors=color_args, autopct='%.2f', fontsize=7)
    pcts = [f'{s * 100:.2f}' for s in series.values / series.sum()]
    expected_texts = list(chain.from_iterable(zip(series.index, pcts)))
    _check_text_labels(ax.texts, expected_texts)
    for t in ax.texts:
        assert t.get_fontsize() == 7