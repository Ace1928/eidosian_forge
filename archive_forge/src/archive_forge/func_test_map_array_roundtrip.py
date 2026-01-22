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
def test_map_array_roundtrip(self):
    data = [[(b'a', 1), (b'b', 2)], [(b'c', 3)], [(b'd', 4), (b'e', 5), (b'f', 6)], [(b'g', 7)]]
    df = pd.DataFrame({'map': data})
    schema = pa.schema([('map', pa.map_(pa.binary(), pa.int32()))])
    _check_pandas_roundtrip(df, schema=schema)