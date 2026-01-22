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
def test_unordered_ts(self):
    index = [date(2012, 10, 1), date(2012, 9, 1), date(2012, 8, 1)]
    values = [3.0, 2.0, 1.0]
    df = DataFrame(np.array(values), index=index, columns=['test'])
    ax = df.plot()
    xticks = ax.lines[0].get_xdata()
    tm.assert_numpy_array_equal(xticks, np.array(index, dtype=object))
    ydata = ax.lines[0].get_ydata()
    tm.assert_numpy_array_equal(ydata, np.array(values))
    xticks = ax.xaxis.get_ticklabels()
    xlocs = [x.get_position()[0] for x in xticks]
    assert Index(xlocs).is_monotonic_increasing
    xlabels = [x.get_text() for x in xticks]
    assert pd.to_datetime(xlabels, format='%Y-%m-%d').is_monotonic_increasing