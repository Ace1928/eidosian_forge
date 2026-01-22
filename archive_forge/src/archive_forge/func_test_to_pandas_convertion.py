import datetime
import itertools
from unittest import mock
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
import modin.pandas as pd
from modin.config import (
from modin.core.dataframe.algebra.default2pandas.groupby import GroupBy
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.pandas.io import from_pandas
from modin.pandas.utils import is_scalar
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import (
from .utils import (
@pytest.mark.skip('Pandas raises a ValueError on empty dictionary aggregation since 1.2.0' + "It's unclear is that was made on purpose or it is a bug. That question" + 'was asked in https://github.com/pandas-dev/pandas/issues/39609.' + 'So until the answer this test is disabled.')
@pytest.mark.parametrize('kwargs', [{'Max': ('cnt', np.max), 'Sum': ('cnt', np.sum), 'Num': ('c', pd.Series.nunique), 'Num1': ('c', pandas.Series.nunique)}, {'func': {'Max': ('cnt', np.max), 'Sum': ('cnt', np.sum), 'Num': ('c', pd.Series.nunique), 'Num1': ('c', pandas.Series.nunique)}}])
def test_to_pandas_convertion(kwargs):
    data = {'a': [1, 2], 'b': [3, 4], 'c': [5, 6]}
    by = ['a', 'b']
    eval_aggregation(*create_test_dfs(data), by=by, **kwargs)