import matplotlib
import numpy as np
import pandas
import pytest
from pandas.core.dtypes.common import is_list_like
import modin.pandas as pd
from modin.config import MinPartitionSize, NPartitions
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('engine', ['python', 'numexpr'])
def test_query_after_insert(engine):
    modin_df = pd.DataFrame({'x': [-1, 0, 1, None], 'y': [1, 2, None, 3]})
    modin_df['z'] = modin_df.eval('x / y')
    modin_df = modin_df.query('z >= 0', engine=engine)
    modin_result = modin_df.reset_index(drop=True)
    modin_result.columns = ['a', 'b', 'c']
    pandas_df = pd.DataFrame({'x': [-1, 0, 1, None], 'y': [1, 2, None, 3]})
    pandas_df['z'] = pandas_df.eval('x / y')
    pandas_df = pandas_df.query('z >= 0', engine=engine)
    pandas_result = pandas_df.reset_index(drop=True)
    pandas_result.columns = ['a', 'b', 'c']
    df_equals(modin_result, pandas_result)
    df_equals(modin_df, pandas_df)