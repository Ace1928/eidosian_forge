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
@pytest.mark.parametrize('method', ['query', 'eval'])
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('local_var', [2])
@pytest.mark.parametrize('engine', ['python', 'numexpr'])
def test_eval_and_query_with_local_and_global_var(method, data, engine, local_var):
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    op = '+' if method == 'eval' else '<'
    for expr in (f'col1 {op} @local_var', f'col1 {op} @TEST_VAR'):
        df_equals(getattr(modin_df, method)(expr, engine=engine), getattr(pandas_df, method)(expr, engine=engine))