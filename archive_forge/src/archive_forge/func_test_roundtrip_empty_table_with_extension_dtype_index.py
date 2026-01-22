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
def test_roundtrip_empty_table_with_extension_dtype_index():
    df = pd.DataFrame(index=pd.interval_range(start=0, end=3))
    table = pa.table(df)
    table.to_pandas().index == pd.Index([{'left': 0, 'right': 1}, {'left': 1, 'right': 2}, {'left': 2, 'right': 3}], dtype='object')