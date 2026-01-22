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
@pytest.mark.large_memory
def test_nested_chunking_valid():

    def roundtrip(df, schema=None):
        tab = pa.Table.from_pandas(df, schema=schema)
        tab.validate(full=True)
        num_chunks = tab.column(0).num_chunks
        assert num_chunks > 1
        tm.assert_frame_equal(tab.to_pandas(self_destruct=True, maps_as_pydicts='strict'), df)
    x = b'0' * 720000000
    roundtrip(pd.DataFrame({'strings': [x, x, x]}))
    struct = {'struct_field': x}
    roundtrip(pd.DataFrame({'structs': [struct, struct, struct]}))
    lists = [x]
    roundtrip(pd.DataFrame({'lists': [lists, lists, lists]}))
    los = [struct]
    roundtrip(pd.DataFrame({'los': [los, los, los]}))
    sol = {'struct_field': lists}
    roundtrip(pd.DataFrame({'sol': [sol, sol, sol]}))
    map_of_los = {'a': los}
    map_type = pa.map_(pa.string(), pa.list_(pa.struct([('struct_field', pa.binary())])))
    schema = pa.schema([('maps', map_type)])
    roundtrip(pd.DataFrame({'maps': [map_of_los, map_of_los, map_of_los]}), schema=schema)