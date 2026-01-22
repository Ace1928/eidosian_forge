import datetime
import io
import warnings
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests.parquet.common import _check_roundtrip
@pytest.mark.pandas
@pytest.mark.parametrize('unit', ['s', 'ms', 'us', 'ns'])
def test_coerce_int96_timestamp_unit(unit):
    i_s = pd.Timestamp('2010-01-01').value / 1000000000
    d_s = np.arange(i_s, i_s + 10, 1, dtype='int64')
    d_ms = d_s * 1000
    d_us = d_ms * 1000
    d_ns = d_us * 1000
    a_s = pa.array(d_s, type=pa.timestamp('s'))
    a_ms = pa.array(d_ms, type=pa.timestamp('ms'))
    a_us = pa.array(d_us, type=pa.timestamp('us'))
    a_ns = pa.array(d_ns, type=pa.timestamp('ns'))
    arrays = {'s': a_s, 'ms': a_ms, 'us': a_us, 'ns': a_ns}
    names = ['ts_s', 'ts_ms', 'ts_us', 'ts_ns']
    table = pa.Table.from_arrays([a_s, a_ms, a_us, a_ns], names)
    expected = pa.Table.from_arrays([arrays.get(unit)] * 4, names)
    read_table_kwargs = {'coerce_int96_timestamp_unit': unit}
    _check_roundtrip(table, expected, read_table_kwargs=read_table_kwargs, use_deprecated_int96_timestamps=True)
    _check_roundtrip(table, expected, version='2.6', read_table_kwargs=read_table_kwargs, use_deprecated_int96_timestamps=True)