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
@pytest.mark.parquet
@pytest.mark.filterwarnings("ignore:Parquet format '2.0':FutureWarning")
def test_timestamp_as_object_parquet(tempdir):
    df = make_df_with_timestamps()
    table = pa.Table.from_pandas(df)
    filename = tempdir / 'timestamps_from_pandas.parquet'
    pq.write_table(table, filename, version='2.0')
    result = pq.read_table(filename)
    df2 = result.to_pandas(timestamp_as_object=True)
    tm.assert_frame_equal(df, df2)