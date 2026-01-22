import json
from textwrap import dedent, indent
from unittest.mock import Mock, patch
import numpy as np
import pandas
import pytest
import modin.pandas as pd
import modin.utils
from modin.error_message import ErrorMessage
from modin.tests.pandas.utils import create_test_dfs
def test_assert_dtypes_equal():
    """Verify that `assert_dtypes_equal` from test utils works correctly (raises an error when it has to)."""
    from modin.tests.pandas.utils import assert_dtypes_equal
    sr1, sr2 = (pd.Series([1.0]), pandas.Series([1.0]))
    assert sr1.dtype == sr2.dtype == 'float'
    assert_dtypes_equal(sr1, sr2)
    sr1 = sr1.astype('int')
    assert sr1.dtype != sr2.dtype and sr1.dtype == 'int'
    assert_dtypes_equal(sr1, sr2)
    sr2 = sr2.astype('str')
    assert sr1.dtype != sr2.dtype and sr2.dtype == 'object'
    with pytest.raises(AssertionError):
        assert_dtypes_equal(sr1, sr2)
    df1, df2 = create_test_dfs({'a': [1], 'b': [1.0]})
    assert_dtypes_equal(df1, df2)
    df1 = df1.astype({'a': 'float'})
    assert df1.dtypes['a'] != df2.dtypes['a']
    assert_dtypes_equal(df1, df2)
    df2 = df2.astype('str')
    with pytest.raises(AssertionError):
        assert_dtypes_equal(sr1, sr2)
    df1 = df1.astype('category')
    df2 = df2.astype('category')
    assert_dtypes_equal(df1, df2)
    df1 = df1.astype({'a': 'str'})
    with pytest.raises(AssertionError):
        assert_dtypes_equal(df1, df2)