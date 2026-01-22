import itertools
import string
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tests.plotting.common import (
from pandas.io.formats.printing import pprint_thing
def test_boxplot_numeric_data(self):
    df = DataFrame({'a': date_range('2012-01-01', periods=100), 'b': np.random.default_rng(2).standard_normal(100), 'c': np.random.default_rng(2).standard_normal(100) + 2, 'd': date_range('2012-01-01', periods=100).astype(str), 'e': date_range('2012-01-01', periods=100, tz='UTC'), 'f': timedelta_range('1 days', periods=100)})
    ax = df.plot(kind='box')
    assert [x.get_text() for x in ax.get_xticklabels()] == ['b', 'c']