import io
import warnings
import matplotlib
import numpy as np
import pandas
import pandas._libs.lib as lib
import pytest
from numpy.testing import assert_array_equal
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('rule', ['5min'])
@pytest.mark.parametrize('closed', ['left', 'right'])
@pytest.mark.parametrize('label', ['right', 'left'])
@pytest.mark.parametrize('on', [None, pytest.param('DateColumn', marks=pytest.mark.xfail(condition=Engine.get() in ('Ray', 'Unidist', 'Dask', 'Python') and StorageFormat.get() != 'Base', reason='https://github.com/modin-project/modin/issues/6399'))])
@pytest.mark.parametrize('level', [None, 1])
def test_resample_specific(rule, closed, label, on, level):
    data, index = (test_data_resample['data'], test_data_resample['index'])
    modin_df = pd.DataFrame(data, index=index)
    pandas_df = pandas.DataFrame(data, index=index)
    if on is None and level is not None:
        index = pandas.MultiIndex.from_product([['a', 'b', 'c', 'd'], pandas.date_range('31/12/2000', periods=len(pandas_df) // 4, freq='h')])
        pandas_df.index = index
        modin_df.index = index
    else:
        level = None
    if on is not None:
        pandas_df[on] = pandas.date_range('22/06/1941', periods=len(pandas_df), freq='min')
        modin_df[on] = pandas.date_range('22/06/1941', periods=len(modin_df), freq='min')
    pandas_resampler = pandas_df.resample(rule, closed=closed, label=label, on=on, level=level)
    modin_resampler = modin_df.resample(rule, closed=closed, label=label, on=on, level=level)
    df_equals(modin_resampler.var(0), pandas_resampler.var(0))
    if on is None and level is None:
        df_equals(modin_resampler.fillna(method='nearest'), pandas_resampler.fillna(method='nearest'))