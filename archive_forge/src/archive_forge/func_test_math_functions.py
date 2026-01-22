import matplotlib
import numpy as np
import pandas
import pytest
import modin.pandas as pd
from modin.config import Engine, NPartitions, StorageFormat
from modin.core.dataframe.pandas.partitioning.axis_partition import (
from modin.tests.pandas.utils import (
from modin.tests.test_utils import warns_that_defaulting_to_pandas
from modin.utils import get_current_execution
@pytest.mark.parametrize('other', [lambda df, axis: 4, lambda df, axis: df.iloc[0] if axis == 'columns' else list(df[df.columns[0]]), lambda df, axis: {label: idx + 1 for idx, label in enumerate(df.axes[0 if axis == 'rows' else 1])}, lambda df, axis: {label if idx % 2 else f'random_key{idx}': idx + 1 for idx, label in enumerate(df.axes[0 if axis == 'rows' else 1][::-1])}], ids=['scalar', 'series_or_list', 'dictionary_keys_equal_columns', 'dictionary_keys_unequal_columns'])
@pytest.mark.parametrize('axis', ['rows', 'columns'])
@pytest.mark.parametrize('op', [*('add', 'radd', 'sub', 'rsub', 'mod', 'rmod', 'pow', 'rpow'), *('truediv', 'rtruediv', 'mul', 'rmul', 'floordiv', 'rfloordiv')])
def test_math_functions(other, axis, op):
    data = test_data['float_nan_data']
    if (op == 'floordiv' or op == 'rfloordiv') and axis == 'rows':
        pytest.xfail(reason='different behavior')
    if op == 'rmod' and axis == 'rows':
        pytest.xfail(reason='different behavior')
    eval_general(*create_test_dfs(data), lambda df: getattr(df, op)(other(df, axis), axis=axis))