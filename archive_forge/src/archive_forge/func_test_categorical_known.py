from __future__ import annotations
import gzip
import os
import warnings
from io import BytesIO, StringIO
from unittest import mock
import pytest
import fsspec
from fsspec.compression import compr
from packaging.version import Version
from tlz import partition_all, valmap
import dask
from dask.base import compute_as_if_collection
from dask.bytes.core import read_bytes
from dask.bytes.utils import compress
from dask.core import flatten
from dask.dataframe._compat import PANDAS_GE_140, PANDAS_GE_200, PANDAS_GE_220, tm
from dask.dataframe.io.csv import (
from dask.dataframe.optimize import optimize_dataframe_getitem
from dask.dataframe.utils import (
from dask.layers import DataFrameIOLayer
from dask.utils import filetext, filetexts, tmpdir, tmpfile
from dask.utils_test import hlg_layer
def test_categorical_known():
    text1 = normalize_text('\n    A,B\n    a,a\n    b,b\n    a,a\n    ')
    text2 = normalize_text('\n    A,B\n    a,a\n    b,b\n    c,c\n    ')
    dtype = pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=False)
    with filetexts({'foo.1.csv': text1, 'foo.2.csv': text2}):
        result = dd.read_csv('foo.*.csv', dtype={'A': 'category', 'B': 'category'})
        assert result.A.cat.known is False
        assert result.B.cat.known is False
        expected = pd.DataFrame({'A': pd.Categorical(['a', 'b', 'a', 'a', 'b', 'c'], categories=dtype.categories), 'B': pd.Categorical(['a', 'b', 'a', 'a', 'b', 'c'], categories=dtype.categories)}, index=[0, 1, 2, 0, 1, 2])
        assert_eq(result, expected)
        result = dd.read_csv('foo.*.csv', dtype={'A': dtype, 'B': 'category'})
        assert result.A.cat.known is True
        assert result.B.cat.known is False
        tm.assert_index_equal(result.A.cat.categories, dtype.categories)
        assert result.A.cat.ordered is False
        assert_eq(result, expected)
        dtype = pd.api.types.CategoricalDtype(['a', 'b', 'c'], ordered=True)
        result = dd.read_csv('foo.*.csv', dtype={'A': dtype, 'B': 'category'})
        expected['A'] = expected['A'].cat.as_ordered()
        assert result.A.cat.known is True
        assert result.B.cat.known is False
        assert result.A.cat.ordered is True
        assert_eq(result, expected)
        result = dd.read_csv('foo.*.csv', dtype=pd.api.types.CategoricalDtype(ordered=False))
        assert result.A.cat.known is False
        result = dd.read_csv('foo.*.csv', dtype='category')
        assert result.A.cat.known is False