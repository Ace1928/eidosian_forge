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
@pytest.mark.parametrize('axis', [0, 1])
@pytest.mark.parametrize('data', test_data_values, ids=test_data_keys)
@pytest.mark.parametrize('func', ['kurt', 'count', 'sum', 'mean', 'all', 'any'])
def test_apply_text_func(data, func, axis):
    func_kwargs = {'axis': axis}
    rows_number = len(next(iter(data.values())))
    level_0 = np.random.choice([0, 1, 2], rows_number)
    level_1 = np.random.choice([3, 4, 5], rows_number)
    index = pd.MultiIndex.from_arrays([level_0, level_1])
    eval_general(*create_test_dfs(data, index=index), lambda df, *args, **kwargs: df.apply(func, *args, **kwargs), **func_kwargs)