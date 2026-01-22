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
def test_subplots_dup_columns_secondary_y_no_subplot(self):
    df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=list('aaaaa'))
    ax = df.plot(secondary_y='a')
    _check_legend_labels(ax, labels=['a (right)'] * 5)
    assert len(ax.lines) == 0
    assert len(ax.right_ax.lines) == 5