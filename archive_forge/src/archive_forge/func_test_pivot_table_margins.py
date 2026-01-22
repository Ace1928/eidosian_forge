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
@pytest.mark.parametrize('data', [test_data['int_data']], ids=['int_data'])
@pytest.mark.parametrize('index', [pytest.param([], id='no_index_cols'), pytest.param(lambda df: df.columns[0], id='single_index_column'), pytest.param(lambda df: [df.columns[0], df.columns[len(df.columns) // 2 - 1]], id='multiple_index_cols')])
@pytest.mark.parametrize('columns', [pytest.param(lambda df: df.columns[len(df.columns) // 2], id='single_column'), pytest.param(lambda df: [*df.columns[len(df.columns) // 2:len(df.columns) // 2 + 4], df.columns[-7]], id='multiple_cols')])
@pytest.mark.parametrize('values', [pytest.param(lambda df: df.columns[-1], id='single_value'), pytest.param(lambda df: df.columns[-4:-1], id='multiple_values')])
@pytest.mark.parametrize('aggfunc', [pytest.param(['mean', 'sum'], id='list_func'), pytest.param(lambda df: {df.columns[5]: 'mean', df.columns[-5]: 'sum'}, id='dict_func')])
@pytest.mark.parametrize('margins_name', [pytest.param('Custom name', id='str_name')])
@pytest.mark.parametrize('fill_value', [None, 0])
def test_pivot_table_margins(data, index, columns, values, aggfunc, margins_name, fill_value, request):
    expected_exception = None
    if 'dict_func' in request.node.callspec.id:
        expected_exception = KeyError("Column(s) ['col28', 'col38'] do not exist")
    eval_general(*create_test_dfs(data), operation=lambda df, *args, **kwargs: df.pivot_table(*args, **kwargs), index=index, columns=columns, values=values, aggfunc=aggfunc, margins=True, margins_name=margins_name, fill_value=fill_value, expected_exception=expected_exception)