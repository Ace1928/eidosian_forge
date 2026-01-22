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
def test_table_from_pandas_keeps_column_order_of_dataframe():
    df1 = pd.DataFrame(OrderedDict([('partition', [0, 0, 1, 1]), ('arrays', [[0, 1, 2], [3, 4], None, None]), ('floats', [None, None, 1.1, 3.3])]))
    df2 = df1[['floats', 'partition', 'arrays']]
    schema1 = pa.schema([('partition', pa.int64()), ('arrays', pa.list_(pa.int64())), ('floats', pa.float64())])
    schema2 = pa.schema([('floats', pa.float64()), ('partition', pa.int64()), ('arrays', pa.list_(pa.int64()))])
    table1 = pa.Table.from_pandas(df1, preserve_index=False)
    table2 = pa.Table.from_pandas(df2, preserve_index=False)
    assert table1.schema.equals(schema1)
    assert table2.schema.equals(schema2)