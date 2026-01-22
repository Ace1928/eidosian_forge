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
def test_roundtrip_map_array_with_pydicts_duplicate_keys():
    keys = pa.array(['foo', 'bar', 'foo'])
    items = pa.array([['a', 'b'], ['c', 'd'], ['1', '2']], pa.list_(pa.string()))
    offsets = [0, 3]
    maps = pa.MapArray.from_arrays(offsets, keys, items)
    ty = pa.map_(pa.string(), pa.list_(pa.string()))
    with pytest.raises(pa.lib.ArrowException):
        maps.to_pandas(maps_as_pydicts='strict')
    series_pydicts = maps.to_pandas(maps_as_pydicts='lossy')
    expected_series_pydicts = pd.Series([{'foo': ['1', '2'], 'bar': ['c', 'd']}])
    assert not maps.equals(pa.Array.from_pandas(series_pydicts, type=ty))
    series_default = maps.to_pandas()
    expected_series_default = pd.Series([[('foo', ['a', 'b']), ('bar', ['c', 'd']), ('foo', ['1', '2'])]])
    assert maps.equals(pa.Array.from_pandas(series_default, type=ty))
    assert len(series_pydicts) == len(expected_series_pydicts)
    for row1, row2 in zip(series_pydicts, expected_series_pydicts):
        assert len(row1) == len(row2)
        for tup1, tup2 in zip(row1.items(), row2.items()):
            assert tup1[0] == tup2[0]
            assert np.array_equal(tup1[1], tup2[1])
    assert len(series_default) == len(expected_series_default)
    for row1, row2 in zip(series_default, expected_series_default):
        assert len(row1) == len(row2)
        for tup1, tup2 in zip(row1, row2):
            assert tup1[0] == tup2[0]
            assert np.array_equal(tup1[1], tup2[1])