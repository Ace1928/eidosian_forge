import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_array_protocol_pandas_extension_types(monkeypatch):
    storage = pa.array([1, 2, 3], type=pa.int64())
    expected = pa.ExtensionArray.from_storage(DummyExtensionType(), storage)
    monkeypatch.setattr(pd.arrays.PeriodArray, '__arrow_array__', PandasArray__arrow_array__, raising=False)
    monkeypatch.setattr(pd.arrays.IntervalArray, '__arrow_array__', PandasArray__arrow_array__, raising=False)
    for arr in [pd.period_range('2012-01-01', periods=3, freq='D').array, pd.interval_range(1, 4).array]:
        result = pa.array(arr)
        assert result.equals(expected)
        result = pa.array(pd.Series(arr))
        assert result.equals(expected)
        result = pa.array(pd.Index(arr))
        assert result.equals(expected)
        result = pa.table(pd.DataFrame({'a': arr})).column('a').chunk(0)
        assert result.equals(expected)