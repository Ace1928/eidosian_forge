from __future__ import annotations
from io import (
from lzma import LZMAError
import os
from tarfile import ReadError
from urllib.error import HTTPError
from xml.etree.ElementTree import ParseError
from zipfile import BadZipFile
import numpy as np
import pytest
from pandas.compat._optional import import_optional_dependency
from pandas.errors import (
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
from pandas.io.common import get_handle
from pandas.io.xml import read_xml
def test_read_xml_nullable_dtypes(parser, string_storage, dtype_backend, using_infer_string):
    data = '<?xml version=\'1.0\' encoding=\'utf-8\'?>\n<data xmlns="http://example.com">\n<row>\n  <a>x</a>\n  <b>1</b>\n  <c>4.0</c>\n  <d>x</d>\n  <e>2</e>\n  <f>4.0</f>\n  <g></g>\n  <h>True</h>\n  <i>False</i>\n</row>\n<row>\n  <a>y</a>\n  <b>2</b>\n  <c>5.0</c>\n  <d></d>\n  <e></e>\n  <f></f>\n  <g></g>\n  <h>False</h>\n  <i></i>\n</row>\n</data>'
    if using_infer_string:
        pa = pytest.importorskip('pyarrow')
        string_array = ArrowStringArrayNumpySemantics(pa.array(['x', 'y']))
        string_array_na = ArrowStringArrayNumpySemantics(pa.array(['x', None]))
    elif string_storage == 'python':
        string_array = StringArray(np.array(['x', 'y'], dtype=np.object_))
        string_array_na = StringArray(np.array(['x', NA], dtype=np.object_))
    elif dtype_backend == 'pyarrow':
        pa = pytest.importorskip('pyarrow')
        from pandas.arrays import ArrowExtensionArray
        string_array = ArrowExtensionArray(pa.array(['x', 'y']))
        string_array_na = ArrowExtensionArray(pa.array(['x', None]))
    else:
        pa = pytest.importorskip('pyarrow')
        string_array = ArrowStringArray(pa.array(['x', 'y']))
        string_array_na = ArrowStringArray(pa.array(['x', None]))
    with pd.option_context('mode.string_storage', string_storage):
        result = read_xml(StringIO(data), parser=parser, dtype_backend=dtype_backend)
    expected = DataFrame({'a': string_array, 'b': Series([1, 2], dtype='Int64'), 'c': Series([4.0, 5.0], dtype='Float64'), 'd': string_array_na, 'e': Series([2, NA], dtype='Int64'), 'f': Series([4.0, NA], dtype='Float64'), 'g': Series([NA, NA], dtype='Int64'), 'h': Series([True, False], dtype='boolean'), 'i': Series([False, NA], dtype='boolean')})
    if dtype_backend == 'pyarrow':
        pa = pytest.importorskip('pyarrow')
        from pandas.arrays import ArrowExtensionArray
        expected = DataFrame({col: ArrowExtensionArray(pa.array(expected[col], from_pandas=True)) for col in expected.columns})
        expected['g'] = ArrowExtensionArray(pa.array([None, None]))
    tm.assert_frame_equal(result, expected)