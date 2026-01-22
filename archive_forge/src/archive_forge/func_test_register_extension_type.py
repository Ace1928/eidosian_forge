from __future__ import annotations
from decimal import Decimal
import pytest
import dask.dataframe as dd
from dask.dataframe.utils import assert_eq
from pandas.tests.extension.decimal.array import DecimalArray, DecimalDtype
from dask.dataframe.extensions import make_array_nonempty, make_scalar
def test_register_extension_type():
    arr = DecimalArray._from_sequence([Decimal('1.0')] * 10)
    ser = pd.Series(arr)
    dser = dd.from_pandas(ser, 2)
    assert_eq(ser, dser)
    df = pd.DataFrame({'A': ser})
    ddf = dd.from_pandas(df, 2)
    assert_eq(df, ddf)