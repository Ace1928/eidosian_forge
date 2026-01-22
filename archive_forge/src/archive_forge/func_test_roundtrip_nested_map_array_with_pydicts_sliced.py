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
def test_roundtrip_nested_map_array_with_pydicts_sliced():
    """
    Slightly more robust test with chunking and slicing
    """
    keys_1 = pa.array(['foo', 'bar'])
    keys_2 = pa.array(['baz', 'qux', 'quux', 'quz'])
    keys_3 = pa.array([], pa.string())
    items_1 = pa.array([['a', 'b'], ['c', 'd']], pa.list_(pa.string()))
    items_2 = pa.array([[], None, [None, 'e'], ['f', 'g']], pa.list_(pa.string()))
    items_3 = pa.array([], pa.list_(pa.string()))
    map_chunk_1 = pa.MapArray.from_arrays([0, 2], keys_1, items_1)
    map_chunk_2 = pa.MapArray.from_arrays([0, 3, 4], keys_2, items_2)
    map_chunk_3 = pa.MapArray.from_arrays([0, 0], keys_3, items_3)
    chunked_array = pa.chunked_array([pa.ListArray.from_arrays([0, 1], map_chunk_1).slice(0), pa.ListArray.from_arrays([0, 1], map_chunk_2.slice(1)).slice(0), pa.ListArray.from_arrays([0, 0], map_chunk_3).slice(0)])
    series_default = chunked_array.to_pandas()
    expected_series_default = pd.Series([[[('foo', ['a', 'b']), ('bar', ['c', 'd'])]], [[('quz', ['f', 'g'])]], []])
    series_pydicts = chunked_array.to_pandas(maps_as_pydicts='strict')
    expected_series_pydicts = pd.Series([[{'foo': ['a', 'b'], 'bar': ['c', 'd']}], [{'quz': ['f', 'g']}], []])
    sliced = chunked_array.slice(1, 3)
    series_default_sliced = sliced.to_pandas()
    expected_series_default_sliced = pd.Series([[[('quz', ['f', 'g'])]], []])
    series_pydicts_sliced = sliced.to_pandas(maps_as_pydicts='strict')
    expected_series_pydicts_sliced = pd.Series([[{'quz': ['f', 'g']}], []])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'elementwise comparison failed', DeprecationWarning)
        tm.assert_series_equal(series_default, expected_series_default)
        tm.assert_series_equal(series_pydicts, expected_series_pydicts)
        tm.assert_series_equal(series_default_sliced, expected_series_default_sliced)
        tm.assert_series_equal(series_pydicts_sliced, expected_series_pydicts_sliced)
    ty = pa.list_(pa.map_(pa.string(), pa.list_(pa.string())))

    def assert_roundtrip(series: pd.Series, data) -> None:
        array_roundtrip = pa.chunked_array(pa.Array.from_pandas(series, type=ty))
        array_roundtrip.validate(full=True)
        assert data.equals(array_roundtrip)
    assert_roundtrip(series_default, chunked_array)
    assert_roundtrip(series_pydicts, chunked_array)
    assert_roundtrip(series_default_sliced, sliced)
    assert_roundtrip(series_pydicts_sliced, sliced)