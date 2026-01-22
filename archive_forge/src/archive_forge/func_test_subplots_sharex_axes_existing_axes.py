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
def test_subplots_sharex_axes_existing_axes(self):
    d = {'A': [1.0, 2.0, 3.0, 4.0], 'B': [4.0, 3.0, 2.0, 1.0], 'C': [5, 1, 3, 4]}
    df = DataFrame(d, index=date_range('2014 10 11', '2014 10 14'))
    axes = df[['A', 'B']].plot(subplots=True)
    df['C'].plot(ax=axes[0], secondary_y=True)
    _check_visible(axes[0].get_xticklabels(), visible=False)
    _check_visible(axes[1].get_xticklabels(), visible=True)
    for ax in axes.ravel():
        _check_visible(ax.get_yticklabels(), visible=True)