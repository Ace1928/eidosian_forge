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
def test_metadata_compat_range_index_pre_0_12():
    a_values = ['foo', 'bar', None, 'baz']
    b_values = ['a', 'a', 'b', 'b']
    a_arrow = pa.array(a_values, type='utf8')
    b_arrow = pa.array(b_values, type='utf8')
    rng_index_arrow = pa.array([0, 2, 4, 6], type='int64')
    gen_name_0 = '__index_level_0__'
    gen_name_1 = '__index_level_1__'
    e1 = pd.DataFrame({'a': a_values}, index=pd.RangeIndex(0, 8, step=2, name='qux'))
    t1 = pa.Table.from_arrays([a_arrow, rng_index_arrow], names=['a', 'qux'])
    t1 = t1.replace_schema_metadata({b'pandas': json.dumps({'index_columns': ['qux'], 'column_indexes': [{'name': None, 'field_name': None, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': {'encoding': 'UTF-8'}}], 'columns': [{'name': 'a', 'field_name': 'a', 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}, {'name': 'qux', 'field_name': 'qux', 'pandas_type': 'int64', 'numpy_type': 'int64', 'metadata': None}], 'pandas_version': '0.23.4'})})
    r1 = t1.to_pandas()
    tm.assert_frame_equal(r1, e1)
    e2 = pd.DataFrame({'qux': a_values}, index=pd.RangeIndex(0, 8, step=2, name='qux'))
    t2 = pa.Table.from_arrays([a_arrow, rng_index_arrow], names=['qux', gen_name_0])
    t2 = t2.replace_schema_metadata({b'pandas': json.dumps({'index_columns': [gen_name_0], 'column_indexes': [{'name': None, 'field_name': None, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': {'encoding': 'UTF-8'}}], 'columns': [{'name': 'a', 'field_name': 'a', 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}, {'name': 'qux', 'field_name': gen_name_0, 'pandas_type': 'int64', 'numpy_type': 'int64', 'metadata': None}], 'pandas_version': '0.23.4'})})
    r2 = t2.to_pandas()
    tm.assert_frame_equal(r2, e2)
    e3 = pd.DataFrame({'a': a_values}, index=pd.RangeIndex(0, 8, step=2, name=None))
    t3 = pa.Table.from_arrays([a_arrow, rng_index_arrow], names=['a', gen_name_0])
    t3 = t3.replace_schema_metadata({b'pandas': json.dumps({'index_columns': [gen_name_0], 'column_indexes': [{'name': None, 'field_name': None, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': {'encoding': 'UTF-8'}}], 'columns': [{'name': 'a', 'field_name': 'a', 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}, {'name': None, 'field_name': gen_name_0, 'pandas_type': 'int64', 'numpy_type': 'int64', 'metadata': None}], 'pandas_version': '0.23.4'})})
    r3 = t3.to_pandas()
    tm.assert_frame_equal(r3, e3)
    e4 = pd.DataFrame({'a': a_values}, index=[pd.RangeIndex(0, 8, step=2, name='qux'), b_values])
    t4 = pa.Table.from_arrays([a_arrow, rng_index_arrow, b_arrow], names=['a', 'qux', gen_name_1])
    t4 = t4.replace_schema_metadata({b'pandas': json.dumps({'index_columns': ['qux', gen_name_1], 'column_indexes': [{'name': None, 'field_name': None, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': {'encoding': 'UTF-8'}}], 'columns': [{'name': 'a', 'field_name': 'a', 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}, {'name': 'qux', 'field_name': 'qux', 'pandas_type': 'int64', 'numpy_type': 'int64', 'metadata': None}, {'name': None, 'field_name': gen_name_1, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}], 'pandas_version': '0.23.4'})})
    r4 = t4.to_pandas()
    tm.assert_frame_equal(r4, e4)
    e5 = pd.DataFrame({'a': a_values}, index=[pd.RangeIndex(0, 8, step=2, name=None), b_values])
    t5 = pa.Table.from_arrays([a_arrow, rng_index_arrow, b_arrow], names=['a', gen_name_0, gen_name_1])
    t5 = t5.replace_schema_metadata({b'pandas': json.dumps({'index_columns': [gen_name_0, gen_name_1], 'column_indexes': [{'name': None, 'field_name': None, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': {'encoding': 'UTF-8'}}], 'columns': [{'name': 'a', 'field_name': 'a', 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}, {'name': None, 'field_name': gen_name_0, 'pandas_type': 'int64', 'numpy_type': 'int64', 'metadata': None}, {'name': None, 'field_name': gen_name_1, 'pandas_type': 'unicode', 'numpy_type': 'object', 'metadata': None}], 'pandas_version': '0.23.4'})})
    r5 = t5.to_pandas()
    tm.assert_frame_equal(r5, e5)