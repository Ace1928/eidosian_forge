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
def test_dictionary_from_pandas_specified_type():
    cat = pd.Categorical.from_codes(np.array([0, 1], dtype='int8'), np.array(['a', 'b'], dtype=object))
    typ = pa.dictionary(index_type=pa.int16(), value_type=pa.string())
    result = pa.array(cat, type=typ)
    assert result.type.equals(typ)
    assert result.to_pylist() == ['a', 'b']
    typ = pa.dictionary(index_type=pa.int8(), value_type=pa.int64())
    with pytest.raises(pa.ArrowInvalid):
        result = pa.array(cat, type=typ)
    typ = pa.dictionary(index_type=pa.int8(), value_type=pa.string(), ordered=True)
    msg = "The 'ordered' flag of the passed categorical values "
    with pytest.raises(ValueError, match=msg):
        result = pa.array(cat, type=typ)
    assert result.to_pylist() == ['a', 'b']
    typ = pa.dictionary(index_type=pa.int16(), value_type=pa.string())
    result = pa.array(cat, type=typ, mask=np.array([False, True]))
    assert result.type.equals(typ)
    assert result.to_pylist() == ['a', None]
    cat = pd.Categorical([])
    typ = pa.dictionary(index_type=pa.int8(), value_type=pa.string())
    result = pa.array(cat, type=typ)
    assert result.type.equals(typ)
    assert result.to_pylist() == []
    typ = pa.dictionary(index_type=pa.int8(), value_type=pa.int64())
    result = pa.array(cat, type=typ)
    assert result.type.equals(typ)
    assert result.to_pylist() == []
    cat = pd.Categorical(['a', 'b'])
    result = pa.array(cat, type=pa.string())
    expected = pa.array(['a', 'b'], type=pa.string())
    assert result.equals(expected)
    assert result.to_pylist() == ['a', 'b']