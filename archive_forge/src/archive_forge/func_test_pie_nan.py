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
def test_pie_nan(self):
    s = Series([1, np.nan, 1, 1])
    _, ax = mpl.pyplot.subplots()
    ax = s.plot.pie(legend=True, ax=ax)
    expected = ['0', '', '2', '3']
    result = [x.get_text() for x in ax.texts]
    assert result == expected