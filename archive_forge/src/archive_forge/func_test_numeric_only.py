import builtins
from io import StringIO
import numpy as np
import pytest
from pandas._libs import lib
from pandas.errors import UnsupportedFunctionCall
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.tests.groupby import get_groupby_method_args
from pandas.util import _test_decorators as td
@pytest.mark.parametrize('kernel, has_arg', [('all', False), ('any', False), ('bfill', False), ('corr', True), ('corrwith', True), ('cov', True), ('cummax', True), ('cummin', True), ('cumprod', True), ('cumsum', True), ('diff', False), ('ffill', False), ('fillna', False), ('first', True), ('idxmax', True), ('idxmin', True), ('last', True), ('max', True), ('mean', True), ('median', True), ('min', True), ('nth', False), ('nunique', False), ('pct_change', False), ('prod', True), ('quantile', True), ('sem', True), ('skew', True), ('std', True), ('sum', True), ('var', True)])
@pytest.mark.parametrize('numeric_only', [True, False, lib.no_default])
@pytest.mark.parametrize('keys', [['a1'], ['a1', 'a2']])
def test_numeric_only(kernel, has_arg, numeric_only, keys):
    df = DataFrame({'a1': [1, 1], 'a2': [2, 2], 'a3': [5, 6], 'b': 2 * [object]})
    args = get_groupby_method_args(kernel, df)
    kwargs = {} if numeric_only is lib.no_default else {'numeric_only': numeric_only}
    gb = df.groupby(keys)
    method = getattr(gb, kernel)
    if has_arg and numeric_only is True:
        result = method(*args, **kwargs)
        assert 'b' not in result.columns
    elif kernel in ('first', 'last') or (kernel in ('any', 'all', 'bfill', 'ffill', 'fillna', 'nth', 'nunique') and numeric_only is lib.no_default):
        result = method(*args, **kwargs)
        assert 'b' in result.columns
    elif has_arg or kernel in ('idxmax', 'idxmin'):
        assert numeric_only is not True
        exception = NotImplementedError if kernel.startswith('cum') else TypeError
        msg = '|'.join(['not allowed for this dtype', 'must be a string or a number', "cannot be performed against 'object' dtypes", 'must be a string or a real number', 'unsupported operand type', 'not supported between instances of', 'function is not implemented for this dtype'])
        with pytest.raises(exception, match=msg):
            method(*args, **kwargs)
    elif not has_arg and numeric_only is not lib.no_default:
        with pytest.raises(TypeError, match="got an unexpected keyword argument 'numeric_only'"):
            method(*args, **kwargs)
    else:
        assert kernel in ('diff', 'pct_change')
        assert numeric_only is lib.no_default
        with pytest.raises(TypeError, match='unsupported operand type'):
            method(*args, **kwargs)