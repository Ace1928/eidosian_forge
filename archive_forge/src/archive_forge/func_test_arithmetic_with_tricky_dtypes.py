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
@pytest.mark.parametrize('op', [*('add', 'radd', 'sub', 'rsub', 'mod', 'rmod', 'pow', 'rpow'), *('truediv', 'rtruediv', 'mul', 'rmul', 'floordiv', 'rfloordiv')])
@pytest.mark.parametrize('val1', [pytest.param([10, 20], id='int'), pytest.param([10, True], id='obj'), pytest.param([True, True], id='bool', marks=pytest.mark.skipif(condition=Engine.get() == 'Native', reason='Fails on HDK')), pytest.param([3.5, 4.5], id='float')])
@pytest.mark.parametrize('val2', [pytest.param([10, 20], id='int'), pytest.param([10, True], id='obj'), pytest.param([True, True], id='bool', marks=pytest.mark.skipif(condition=Engine.get() == 'Native', reason='Fails on HDK')), pytest.param([3.5, 4.5], id='float'), pytest.param(2, id='int scalar'), pytest.param(True, id='bool scalar', marks=pytest.mark.skipif(condition=Engine.get() == 'Native', reason='Fails on HDK')), pytest.param(3.5, id='float scalar')])
def test_arithmetic_with_tricky_dtypes(val1, val2, op, request):
    modin_df1, pandas_df1 = create_test_dfs(val1)
    modin_df2, pandas_df2 = create_test_dfs(val2) if isinstance(val2, list) else (val2, val2)
    expected_exception = None
    if ('bool-bool' in request.node.callspec.id or 'bool scalar-bool' in request.node.callspec.id) and op in ['pow', 'rpow', 'truediv', 'rtruediv', 'floordiv', 'rfloordiv']:
        op_name = op[1:] if op.startswith('r') else op
        expected_exception = NotImplementedError(f"operator '{op_name}' not implemented for bool dtypes")
    elif ('bool-bool' in request.node.callspec.id or 'bool scalar-bool' in request.node.callspec.id) and op in ['sub', 'rsub']:
        expected_exception = TypeError('numpy boolean subtract, the `-` operator, is not supported, ' + 'use the bitwise_xor, the `^` operator, or the logical_xor function instead.')
    eval_general((modin_df1, modin_df2), (pandas_df1, pandas_df2), lambda dfs: getattr(dfs[0], op)(dfs[1]), expected_exception=expected_exception)