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
@pytest.mark.parametrize('other', [lambda df: df[:-2 ** 4], lambda df: df[df.columns[0]].reset_index(drop=True)], ids=['check_missing_value', 'check_different_index'])
@pytest.mark.parametrize('fill_value', [None, 3.0])
@pytest.mark.parametrize('op', [*('add', 'radd', 'sub', 'rsub', 'mod', 'rmod', 'pow', 'rpow'), *('truediv', 'rtruediv', 'mul', 'rmul', 'floordiv', 'rfloordiv')])
def test_math_functions_fill_value(other, fill_value, op, request):
    data = test_data['int_data']
    modin_df, pandas_df = (pd.DataFrame(data), pandas.DataFrame(data))
    expected_exception = None
    if 'check_different_index' in request.node.callspec.id and fill_value == 3.0:
        expected_exception = NotImplementedError('fill_value 3.0 not supported.')
    eval_general(modin_df, pandas_df, lambda df: getattr(df, op)(other(df), axis=0, fill_value=fill_value), expected_exception=expected_exception, comparator_kwargs={'check_dtypes': get_current_execution() != 'BaseOnPython'})