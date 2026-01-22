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
@pytest.mark.parametrize('groupby_axis', [lib.no_default, 1])
@pytest.mark.parametrize('shift_axis', [lib.no_default, 1])
@pytest.mark.parametrize('groupby_sort', [True, False])
def test_shift_freq(groupby_axis, shift_axis, groupby_sort):
    pandas_df = pandas.DataFrame({'col1': [1, 0, 2, 3], 'col2': [4, 5, np.NaN, 7], 'col3': [np.NaN, np.NaN, 12, 10], 'col4': [17, 13, 16, 15]})
    modin_df = from_pandas(pandas_df)
    new_index = pandas.date_range('1/12/2020', periods=4, freq='s')
    if groupby_axis == 0 and shift_axis == 0:
        pandas_df.index = modin_df.index = new_index
        by = [['col2', 'col3'], ['col2'], ['col4'], [0, 1, 0, 2]]
    else:
        pandas_df.index = modin_df.index = new_index
        pandas_df.columns = modin_df.columns = new_index
        by = [[0, 1, 0, 2]]
    for _by in by:
        pandas_groupby = pandas_df.groupby(by=_by, axis=groupby_axis, sort=groupby_sort)
        modin_groupby = modin_df.groupby(by=_by, axis=groupby_axis, sort=groupby_sort)
        eval_general(modin_groupby, pandas_groupby, lambda groupby: groupby.shift(axis=shift_axis, freq='s'))