import string
import numpy as np
import pytest
from pandas.compat import is_platform_linux
from pandas.compat.numpy import np_version_gte1p24
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_subplots_ts_share_axes(self):
    _, axes = mpl.pyplot.subplots(3, 3, sharex=True, sharey=True)
    mpl.pyplot.subplots_adjust(left=0.05, right=0.95, hspace=0.3, wspace=0.3)
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 9)), index=date_range(start='2014-07-01', freq='ME', periods=10))
    for i, ax in enumerate(axes.ravel()):
        df[i].plot(ax=ax, fontsize=5)
    for ax in axes[0:-1].ravel():
        _check_visible(ax.get_xticklabels(), visible=False)
    for ax in axes[-1].ravel():
        _check_visible(ax.get_xticklabels(), visible=True)
    for ax in axes[[0, 1, 2], [0]].ravel():
        _check_visible(ax.get_yticklabels(), visible=True)
    for ax in axes[[0, 1, 2], [1]].ravel():
        _check_visible(ax.get_yticklabels(), visible=False)
    for ax in axes[[0, 1, 2], [2]].ravel():
        _check_visible(ax.get_yticklabels(), visible=False)