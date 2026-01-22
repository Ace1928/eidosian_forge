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
@pytest.mark.parametrize('idx', [Index, pd.CategoricalIndex])
def test_bar_categorical(self, idx):
    df = DataFrame(np.random.default_rng(2).standard_normal((6, 5)), index=idx(list('ABCDEF')), columns=idx(list('abcde')))
    ax = df.plot.bar()
    ticks = ax.xaxis.get_ticklocs()
    tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
    assert ax.get_xlim() == (-0.5, 5.5)
    assert ax.patches[0].get_x() == -0.25
    assert ax.patches[-1].get_x() == 5.15
    ax = df.plot.bar(stacked=True)
    tm.assert_numpy_array_equal(ticks, np.array([0, 1, 2, 3, 4, 5]))
    assert ax.get_xlim() == (-0.5, 5.5)
    assert ax.patches[0].get_x() == -0.25
    assert ax.patches[-1].get_x() == 4.75