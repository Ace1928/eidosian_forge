import warnings
import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, RangePartitioning, StorageFormat
from modin.pandas.io import to_pandas
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
@pytest.mark.skipif(StorageFormat.get() != 'Pandas', reason='We only need to test this case where sort does not default to pandas.')
@pytest.mark.parametrize('ascending', [True, False], ids=['True', 'False'])
@pytest.mark.parametrize('na_position', ['first', 'last'], ids=['first', 'last'])
def test_sort_values_with_only_one_non_na_row_in_partition(ascending, na_position):
    pandas_df = pandas.DataFrame(np.random.rand(1000, 100), columns=[f'col {i}' for i in range(100)])
    pandas_df.iloc[340:] = np.NaN
    pandas_df.iloc[-1] = -4.0
    modin_df = pd.DataFrame(pandas_df)
    eval_general(modin_df, pandas_df, lambda df: df.sort_values('col 3', ascending=ascending, na_position=na_position))