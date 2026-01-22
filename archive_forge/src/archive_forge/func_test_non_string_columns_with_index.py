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
def test_non_string_columns_with_index(self):
    df = pd.DataFrame({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0, 6.0]})
    df = df.set_index(0)
    with pytest.warns(UserWarning):
        table = pa.Table.from_pandas(df)
        assert table.field(0).name == '1'
    expected = df.copy()
    expected.index.name = str(expected.index.name)
    with pytest.warns(UserWarning):
        _check_pandas_roundtrip(df, expected=expected, preserve_index=True)